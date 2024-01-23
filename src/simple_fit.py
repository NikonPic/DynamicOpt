# %%
"""
Goals:
Manage all params with hydra
Manage multiple environments
Use Soft Actor Critic to identify the parameters
"""
import copy
from datetime import datetime
import os
import time
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import torch.multiprocessing as mp
import yaml
# local imports
from config import OverallSettings
from model_training_modules import ParamLearner, SACDeepCritic, flatten_dict
from model_simulation import DiskModel, HandModel, KneeModel, MeasureDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from functools import wraps


cs = ConfigStore.instance()
cs.store(name='ppo_config', node=OverallSettings)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} s.")
        return result
    return wrapper


def create_env(cfg: OverallSettings):
    """Create an environment using the KneeModel class based on the provided configuration"""
    # Create dataset
    dset = MeasureDataset(cfg)

    if cfg.env.model_type == 'knee':
        # Create environment
        env = KneeModel(
            cfg.env,
            dataset=dset,
        )
    elif cfg.env.model_type == 'hand':
        env = HandModel(
            cfg.env,
            dataset=dset,
        )
    else:
        env = DiskModel(
            cfg.env,
            dataset=dset,
        )

    all_state_len = len(env.complete_state())

    return env, all_state_len


def create_ac(cfg: OverallSettings):
    """Create an actor and critic pair based on the provided configuration and state length"""
    # Create actor
    actor = ParamLearner(
        width=cfg.actor.width,
        use_std=cfg.actor.use_std,
        reinforce=cfg.actor.reinforce,
        param_file=f'{cfg.env.param_path}/{cfg.env.param_file}'
    )
    return actor


# %% data strucuture


@dataclass
class TransitionClass:
    action: np.array
    reward: np.array


class TransitionBatch:
    """Class to handle a batch of transition instances."""

    def __init__(self, translist, cuda_device=None):
        self.translist = translist
        self.len = len(translist)
        self.cuda_device = torch.device(
            f"cuda:{cuda_device}") if cuda_device is not None else None
        self.fill_class_fields()

    def fill_class_fields(self):
        """
        Fill class fields with stacked tensors of corresponding attributes
        from the list of transition instances.
        """
        for idx, fieldname in enumerate(TransitionClass.__dataclass_fields__):
            setattr(self, fieldname, self.translist[idx].to(self.cuda_device))


def normalize_states(states, state_mean, state_std):
    # Add a small constant to avoid division by zero
    return (states - state_mean) / (state_std + 1e-8)


class CollectedDataDataset(Dataset):
    """A dataset class for storing and retrieving transition instances."""

    def __init__(self, transitions: TransitionClass):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions.reward)

    def __getitem__(self, index):
        action = self.transitions.action[index]
        reward = self.transitions.reward[index]
        return (action, reward)


class TransitionBatchDataLoader(DataLoader):
    """A data loader that returns batches of transition instances."""

    def __init__(self, dataset, batch_size, shuffle=True, cuda_device=None, *args, **kwargs):
        self.dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self.cuda_device = cuda_device
        super().__init__(dataset, batch_size=batch_size,
                         shuffle=shuffle, pin_memory=False, *args, **kwargs)

    def __iter__(self):
        """
        Create an iterator that yields batches of transition instances
        as instances of the TransitionBatch class.
        """
        for batch_transitions in super().__iter__():
            yield TransitionBatch(batch_transitions, self.cuda_device)


class RewardScaler:
    def __init__(self):
        self.mean = 0
        self.std = 1
        self.count = 0

    def update(self, reward):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.std = np.sqrt(
            (self.std**2 * (self.count - 1) + delta * delta2) / self.count)

    def normalize(self, reward):
        # Small constant to avoid division by zero
        return (reward - self.mean) / (self.std + 1e-8)


class SACAgent:
    """class to perform the state action logic with the environment"""

    def __init__(self, cfg: OverallSettings, env: KneeModel, device='cpu'):
        self.env = copy.deepcopy(env)
        self.actor = create_ac(cfg)

        self.device = device
        self.actor.to(device)

        # predefine
        self.param_in = torch.ones(1, self.actor.paramlen).to(device)
        self.reward_scaler = RewardScaler()
        self.reward_sum = 0

    def update_ac(self, actor: nn.Module):
        self.actor = pickle.loads(pickle.dumps(actor)).to(self.device)
        self.actor = self.actor.to(self.device)

    def init_lists(self):
        self.batch_action = []
        self.batch_reward = []

    def collect_data(self, shared_queue: mp.Queue, timesteps_counter: mp.Value, lock: mp.Lock, reward_q: mp.Queue, eplen_q: mp.Queue, max_timesteps: int):
        """collect new data until the queue is filled"""
        keep_collecting = True
        self.max_timesteps = max_timesteps
        self.episode_lenghts = []
        self.episode_rewards = []

        while keep_collecting:
            self.run_episode()
            self.episode_lenghts.append(len(self.batch_reward))
            self.episode_rewards.append(np.mean(self.batch_reward))
            keep_collecting = self.put_data_to_queue(
                shared_queue, timesteps_counter, lock)

        self.avg_episode_reward = np.mean(self.episode_rewards)
        self.avg_episode_length = np.mean(self.episode_lenghts)

        reward_q.put(self.avg_episode_reward)
        eplen_q.put(self.avg_episode_length)

    @torch.no_grad()
    def run_episode(self):
        """complete run of single episode"""
        self.actor.eval()

        _, action, _, parameters = self.actor(self.param_in)
        action = action[0].detach().numpy()
        parameters = parameters[0]
        self.env.reset_new(parameters)
        self.init_lists()
        done = False
        reward_sum = 0
        tstep = 1

        while not done:
            # simulate one step
            _, reward, done = self.env.step()
            reward_sum += reward
            tstep += 1

        self.batch_action.append(action)
        self.batch_reward.append(reward_sum / tstep)

    def put_data_to_queue(self, shared_queue: mp.Queue, timesteps_counter: mp.Value, lock: mp.Lock):
        # Convert the entire batch of data to NumPy arrays before the loop
        batch_action_np = np.stack(self.batch_action)
        batch_reward_np = np.stack(self.batch_reward)

        train_data = zip(batch_action_np, batch_reward_np)

        for action, reward in train_data:
            state_transition = TransitionClass(action, reward)

            # Add data to the queue if it is not full yet
            if not self.check_and_put_queue(shared_queue, timesteps_counter, lock, state_transition):
                return False

        return True

    def check_and_put_queue(self, shared_queue: mp.Queue, timesteps_counter: mp.Value, lock: mp.Lock, state_transition: TransitionClass):
        """this function checks if the episode can be fitted and further increases the value of the counter"""
        with lock:
            if timesteps_counter.value >= self.max_timesteps:
                self.keep_collecting = False
                return False
            else:
                timesteps_counter.value += 1

        shared_queue.put(state_transition)
        return True

    def return_single_state_transition(self):
        self.run_episode()
        action = self.batch_action[0]
        reward = self.batch_reward[0]

        state_transition = TransitionClass(action, reward)
        return state_transition

    def log_image_progress(self):
        """save an image to the session"""
        # get the validation as well
        self.actor.eval()
        _, _, _, parameters = self.actor(self.param_in, sample=False)
        parameters = parameters[0]
        return self.env.perform_all(parameters)


class ValueSchedule(nn.Module):

    def __init__(self, cfg: OverallSettings) -> None:
        super(ValueSchedule, self).__init__()
        self.cfg: OverallSettings = cfg
        set_seed(cfg.general.seed)

        self.clip_ratio = self.cfg.simple.clip_ratio
        self.num_agents = cfg.general.num_workers
        self.num_epochs = 0
        self.steps_per_epoch = cfg.simple.steps_per_epoch

        # multiprocessing
        self.manager = mp.Manager()
        self.shared_list = mp.Queue()
        self.timestep_counter = mp.Value('i', 0)
        self.lock = mp.Lock()
        self.rewards = mp.Queue()
        self.ep_lengths = mp.Queue()
        self.lam_r = self.cfg.env.lam_reward

        # create ac for training
        env, _ = create_env(cfg)
        self.actor = create_ac(cfg)
        self.actor_old = create_ac(cfg)

        # setup the distributed agents
        self.agents = [SACAgent(cfg, env)
                       for _ in tqdm(range(self.num_agents))]

        self.cuda_device = cfg.general.cuda_device
        self.device = torch.device(f"cuda:{cfg.general.cuda_device}")
        self.empty_t = torch.zeros(0).to(self.device)
        self.actor.cuda(self.cuda_device)

        self.mse = torch.nn.MSELoss()
        self.configure_optimizers()

        # logging
        now = datetime.now()
        self.dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        logdir = f'./src/trained_models/{self.dt_string}'
        os.mkdir(logdir)
        self.writer = SummaryWriter(log_dir=logdir)

        self.param_in = torch.ones(
            cfg.simple.batch_size, len(self.actor.paramlist)).cuda(self.cuda_device).float()

        # synchronize agents
        self.update_agents()

        # preallocate
        self.memory_size = cfg.simple.memory_size
        self.transitions = self.preallocate_memory()
        self.queue_counter = 0
        self.all_data_available = False

        # predefine target entropy
        self.target_entropy = - \
            torch.prod(torch.Tensor(len(self.actor.paramlist))).item()

        # handle data gen comparison
        self.param_path = cfg.env.param_path
        self.data_gen = cfg.env.param_data_gen_file
        self.keep_data_gen_loggin = False

        if self.data_gen is not None:
            self.keep_data_gen_loggin = True
            with open(f'{self.param_path}/{self.data_gen}', "r") as stream:
                self.params_gen = yaml.safe_load(stream)
            self.params_gen = np.array([float(x)
                                       for x in flatten_dict(self.params_gen)])

        # prefill the queue
        self.collect_data(self.cfg.simple.steps_per_epoch)

    def preallocate_memory(self, mydtype=np.float32):
        """for faster dataset creation"""
        dummy_transition = self.agents[0].return_single_state_transition()

        # Compute shapes for the preallocated arrays
        shapes = {
            attr: (self.memory_size,) +
            getattr(dummy_transition, attr).shape
            for attr in ('action', 'reward')
        }

        # Preallocate arrays with the computed shapes and specified data type
        preallocated_data = {
            attr: np.zeros(shape, dtype=mydtype)
            for attr, shape in shapes.items()
        }

        # Create and return a TransitionClass instance with preallocated arrays
        return TransitionClass(**preallocated_data)

    def setup_dis_processes(self, collect_number: int):
        """setup and start the dis processes"""
        self.processes = []
        # setup the distributed agents
        for agent in self.agents:
            p = mp.Process(
                target=agent.collect_data,
                args=(
                    self.shared_list,
                    self.timestep_counter,
                    self.lock,
                    self.rewards,
                    self.ep_lengths,
                    collect_number,
                ),
            )
            p.start()
            self.processes.append(p)

    def configure_optimizers(self):
        """ Initialize Adam optimizers for the actor, critic, and alpha """
        self.optimizer_actor = optim.Adam(
            self.actor.parameters(), lr=self.cfg.actor.lr)

    @timeit
    def collect_data(self, collect_number: int):
        """distributed data collection using distributed proccesses"""

        self.timestep_counter = mp.Value('i', 0)
        # setup
        self.setup_dis_processes(collect_number)
        self.collect_data_from_queue(self.shared_list, collect_number)

        # wait for completion
        for p in self.processes:
            p.join()

    def create_dataloader(self):
        self.calculate_mean_reward()
        if self.all_data_available:
            # recreate the dataloader with the new collected data
            self.dataloader = TransitionBatchDataLoader(
                CollectedDataDataset(self.transitions),
                batch_size=self.cfg.simple.batch_size,
                shuffle=True,
                num_workers=self.cfg.general.num_trainers,
                cuda_device=self.cuda_device
            )
        else:
            sub_actions = self.transitions.action[:self.queue_counter]
            sub_rewards = self.transitions.reward[:self.queue_counter]
            sub_transitions = TransitionClass(sub_actions, sub_rewards)
            print(len(sub_actions))

            self.dataloader = TransitionBatchDataLoader(
                CollectedDataDataset(sub_transitions),
                batch_size=self.cfg.simple.batch_size,
                shuffle=True,
                num_workers=self.cfg.general.num_trainers,
                cuda_device=self.cuda_device
            )

        self.download_ep_info_from_queue()

    def collect_data_from_queue(self, shared_queue: mp.Queue, collect_number: int):
        """Constantly unload the queue and add to dataset for SAC"""
        counter = 0
        queue_counter = self.queue_counter
        max_len = self.memory_size

        # Store references to the arrays in variables to minimize array indexing
        action_arr = self.transitions.action
        reward_arr = self.transitions.reward

        # Refactor loop structure to avoid nested loops
        while counter < collect_number:

            try:
                data = shared_queue.get(block=True, timeout=5)
                action_arr[queue_counter, :] = data.action
                reward_arr[queue_counter] = data.reward

                counter += 1
                queue_counter += 1

                if queue_counter >= max_len:
                    queue_counter = 0
                    self.all_data_available = True

            except:
                print('Queue empty')
                time.sleep(0.05)

        self.queue_counter = queue_counter

    def download_ep_info_from_queue(self):
        """take the avg data from the queue"""
        rewards = []
        while not self.rewards.empty():
            rewards.append(self.rewards.get())
        self.avg_rewards = np.mean(rewards)

        ep_lengths = []
        while not self.ep_lengths.empty():
            ep_lengths.append(self.ep_lengths.get())
        self.avg_ep_lengths = np.mean(ep_lengths)

    @timeit
    def update_agents(self):
        """update the distributed agents"""
        for agent in self.agents:
            agent.update_ac(self.actor)

        self.actor_old = pickle.loads(pickle.dumps(self.actor)).to(self.device)

    @timeit
    def train_epoch(self):
        """training including a whole epoche"""
        self.total_loss_a = 0
        self.num_batches = 0

        for _ in range(self.cfg.simple.nb_optim_iters):
            self.train_full_dataset_policy()

    def train_full_dataset_policy(self):
        self.optimizer_actor.zero_grad()
        for batch_id, batch in enumerate(self.dataloader):
            self.optimizer_actor.zero_grad()
            loss_a = self.actor_policy_loss(batch)
            loss_a.backward()

            # Apply gradient clipping to the actor
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.cfg.actor.max_grad_norm)
            self.optimizer_actor.step()

            self.total_loss_a += loss_a.item()
            self.num_batches += 1

            if self.cfg.simple.nb_batch_ids > 0 and batch_id > self.cfg.simple.nb_batch_ids:
                break

    def calculate_mean_reward(self):
        # first get all relevant transitions:
        if self.all_data_available:
            # use whole dataset
            rewards = self.transitions.reward
        else:
            rewards = self.transitions.reward[:self.queue_counter]

        # get the mean of the rewards
        self.reward_mean = torch.tensor(np.mean(rewards)).to(self.device)
        self.reward_std = torch.tensor(np.std(rewards)).to(self.device)

    def actor_policy_loss(self, batch: TransitionBatch):
        adv = (batch.reward - self.reward_mean) / self.reward_std
        batch_size = adv.shape[0]
        param_in = self.param_in[:batch_size]

        pi = self.actor.forward_fast(param_in)
        logp = self.actor.get_log_prob(pi, batch.action)

        with torch.no_grad():
            pi_old = self.actor_old.forward_fast(param_in)
            logp_old = self.actor_old.get_log_prob(pi_old, batch.action)

        # Clipping the log probability differences to maintain numerical stability
        logp_diff = logp - logp_old
        logp_diff_clipped = torch.clamp(logp_diff, min=-10, max=10)

        ratio = torch.exp(logp_diff_clipped)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio,
                               1 + self.clip_ratio) * adv.float()
        loss = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss

    def run_training(self):
        """run the whole training"""
        for epoch in tqdm(range(self.cfg.general.epochs)):
            self.epoch = epoch
            self.collect_data(collect_number=self.cfg.simple.steps_per_epoch)
            self.create_dataloader()
            self.train_epoch()
            self.update_agents()
            self.log_progress()

     # functions for logging
    # ---------------------------------------------------------------------------------------------------
    @timeit
    def log_progress(self):
        avg_loss_a = self.total_loss_a / self.num_batches

        self.writer.add_scalar(
            "ALLG/avg_ep_len", float(self.avg_ep_lengths), self.epoch)
        self.writer.add_scalar(
            "ALLG/avg_ep_loss", float((1 - self.avg_rewards) / self.lam_r), self.epoch)
        self.writer.add_scalar(
            "ALLG/loss_actor", float(avg_loss_a), self.epoch)

        self.log_params()
        self.log_figure()
        self.save()

    def log_params(self):
        """log the progression of all individual parameters"""
        if self.epoch % self.cfg.log.param_intervall:
            return

        self.actor.eval()
        _, _, _, parameters = self.actor(self.param_in, sample=False)
        parameters = parameters[0]
        self.actor.train()
        self.log_parameters(parameters)

    def log_parameters(self, parameters):
        """recur"""
        def add_scalar(prefix, value):
            self.writer.add_scalar(prefix, float(value), self.epoch)

        def traverse(prefix, items):
            for key, value in items.items():
                new_prefix = f'{prefix}/{key}'
                if isinstance(value, dict):
                    traverse(new_prefix, value)
                else:
                    add_scalar(new_prefix, value)

        for key, sub_params in parameters.items():
            traverse(f'Parameters/{key}', sub_params)

        if self.keep_data_gen_loggin:
            cur_flatten = np.array([float(x)
                                   for x in flatten_dict(parameters)])
            mse = np.mean((cur_flatten - self.params_gen)**2)
            self.writer.add_scalar('MSE-Params', float(mse), self.epoch)

    def log_figure(self):
        """log current progress in image"""
        if self.epoch % self.cfg.log.figure_intervall:
            return
        fig, err = self.agents[0].log_image_progress()
        fig.savefig(
            f'./src/trained_models/{self.dt_string}/fig_{self.epoch}.png')
        self.writer.add_scalar(
            "ALLG/val_err", float(err), self.epoch)

    def save(self):
        """save the state of the trained model"""
        if self.epoch % self.cfg.log.param_save_intervall:
            return

        self.actor.eval()
        _, _, _, parameters = self.actor(self.param_in, sample=False)
        parameters = parameters[0]

        with open(f'./src/trained_models/{self.dt_string}/parameters_{self.epoch}.yaml', 'w') as f:
            yaml.dump(parameters, f)
    # ---------------------------------------------------------------------------------------------------


# %%


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: OverallSettings):
    trainer = ValueSchedule(cfg)
    trainer.run_training()


# %%
if __name__ == '__main__':
    main()

# %%
