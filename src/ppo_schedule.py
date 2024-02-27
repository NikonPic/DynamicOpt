# %%
"""
Goals:
Manage all params with hydra
Manage multiple environments
Use Proximal Policy Optimization to find the unidentified parameters
"""
import copy
from datetime import datetime
import os
import time
from typing import List
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
from model_training_modules import Critic, ParamLearner, flatten_dict
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

# %% data strucuture


@dataclass
class TransitionClass:
    state: np.array
    action: np.array
    logp_old: np.array
    qval: np.array
    adv: np.array


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


class CollectedDataDataset(Dataset):
    """A dataset class for storing and retrieving transition instances."""

    def __init__(self, transitions: TransitionClass):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions.adv)

    def __getitem__(self, index):
        state = self.transitions.state[index, :]
        action = self.transitions.action[index, :]
        logp = self.transitions.logp_old[index, :]
        qval = self.transitions.qval[index]
        adv = self.transitions.adv[index]
        return (state, action, logp, qval, adv)


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

# %%


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


def create_ac(cfg: OverallSettings, all_state_len: int):
    """Create an actor and critic pair based on the provided configuration and state length"""
    # Create actor
    actor = ParamLearner(
        width=cfg.actor.width,
        use_std=cfg.actor.use_std,
        reinforce=cfg.actor.reinforce,
        param_file=f'{cfg.env.param_path}/{cfg.env.param_file}'
    )

    # Create critic
    critic = Critic(
        width=cfg.critic.width,
        input_len=all_state_len,
        output_len=1
    )
    return actor, critic


class PPOAgent:
    """class to perform the state action logic with the environment"""

    def __init__(self, cfg: OverallSettings, env: KneeModel, all_state_len, device='cpu'):
        self.gamma = cfg.ppo.gamma
        self.lam = cfg.ppo.lam

        self.env = copy.deepcopy(env)
        self.actor, self.critic = create_ac(cfg, all_state_len)

        self.device = device
        self.actor.to(device)
        self.critic.to(device)

        self.steps_per_epoch = cfg.ppo.steps_per_epoch

        # predefine
        self.param_in = torch.ones(1, self.actor.paramlen).to(device)

    def update_ac(self, actor: nn.Module, critic: nn.Module, epoch_fac=1):
        self.critic = pickle.loads(pickle.dumps(critic))
        self.critic = self.critic.to(self.device)
        
        self.actor = pickle.loads(pickle.dumps(actor)).to(self.device)
        self.actor = self.actor.to(self.device)
        self.actor.epoch_fac = epoch_fac

    def discount_rewards(self, rewards: List[float], discount: float) -> List[float]:
        """Calculate the discounted rewards of all rewards in list"""
        assert isinstance(rewards[0], float)

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * discount) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def calc_advantage(self, rewards: List[float], values: List[float], last_value: float) -> List[float]:
        """Calculate the advantage given rewards, state values, and the last value of episode"""
        rews = rewards + [last_value]
        vals = values + [last_value]
        # GAE
        delta = [rews[i] + self.gamma * vals[i + 1] - vals[i]
                 for i in range(len(rews) - 1)]
        adv = self.discount_rewards(delta, self.gamma * self.lam)
        return adv

    def init_lists(self):
        self.batch_state = []
        self.batch_action = []
        self.batch_logp = []
        self.batch_adv = []
        self.batch_qval = []

        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []

    def collect_data(self, shared_queue: mp.Queue, timesteps_counter: mp.Value, lock: mp.Lock, reward_q: mp.Queue, eplen_q: mp.Queue):
        """collect new data until the queue is filled"""
        keep_collecting = True
        self.episode_lenghts = []
        self.episode_rewards = []

        while keep_collecting:
            self.run_episode()
            self.episode_lenghts.append(len(self.ep_rewards))
            self.episode_rewards.append(np.mean(self.ep_rewards))
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
        self.critic.eval()

        _, action, logp, parameters = self.actor(self.param_in)
        parameters = parameters[0]
        self.env.reset_new(parameters)
        self.init_lists()

        done = False
        episode_len = 0

        while not done:
            state, reward, done = self.env.step()
            value = self.critic(torch.reshape(state, [1, -1]))
            episode_len += 1

            self.batch_state.append(state.detach().float())
            self.batch_action.append(action[0].detach())
            self.batch_logp.append(logp.detach())
            self.ep_rewards.append(reward)
            self.ep_values.append(value.item())

        earlystop = True if episode_len < self.env.max_sim_len else False

        # decide if last value is to be continued or 0
        if not earlystop:
            last_value = value  # take estimate
        else:
            last_value = -1  # failed, so use bad value here

        # discounted cumulative reward
        self.batch_qval += self.discount_rewards(
            self.ep_rewards + [last_value], self.gamma)[:-1]

        # advantage
        self.batch_adv += self.calc_advantage(
            self.ep_rewards, self.ep_values, last_value)

    def put_data_to_queue(self, shared_queue: mp.Queue, timesteps_counter: mp.Value, lock: mp.Lock):
        # Convert the entire batch of data to NumPy arrays before the loop
        batch_state_np = np.stack(self.batch_state)
        batch_action_np = np.stack(self.batch_action)
        batch_logp_np = np.stack(self.batch_logp)
        batch_qval_np = np.array(self.batch_qval, dtype=np.float32)
        batch_adv_np = np.array(self.batch_adv, dtype=np.float32)

        train_data = zip(batch_state_np, batch_action_np,
                         batch_logp_np, batch_qval_np, batch_adv_np)

        for state, action, logp_old, qval, adv in train_data:
            state_transition = TransitionClass(
                state, action, logp_old, qval, adv)

            # Add data to the queue if it is not full yet
            if not self.check_and_put_queue(shared_queue, timesteps_counter, lock, state_transition):
                return False

        return True

    def check_and_put_queue(self, shared_queue: mp.Queue, timesteps_counter: mp.Value, lock: mp.Lock, state_transition: TransitionClass):
        """this function checks if the episode can be fitted and further increases the value of the counter"""
        with lock:
            if timesteps_counter.value >= self.steps_per_epoch:
                self.keep_collecting = False
                return False
            else:
                timesteps_counter.value += 1

        shared_queue.put(state_transition)
        return True

    def return_single_state_transition(self):
        self.run_episode()

        state = self.batch_state[0].numpy()
        action = self.batch_action[0].numpy()
        logp_old = self.batch_logp[0].numpy()
        qval = np.array(self.batch_qval[0], dtype=np.float32)
        adv = np.array(self.batch_adv[0], dtype=np.float32)

        state_transition = TransitionClass(state, action, logp_old, qval, adv)
        return state_transition

    # some logging
    def log_image_progress(self):
        """save an image to the session"""
        # get the validation as well
        self.actor.eval()
        _, _, _, parameters = self.actor(self.param_in, sample=False)
        parameters = parameters[0]
        return self.env.perform_all(parameters)


class PPOSchedule(nn.Module):

    def __init__(self, cfg: OverallSettings) -> None:
        super(PPOSchedule, self).__init__()
        self.cfg: OverallSettings = cfg
        set_seed(cfg.general.seed)

        self.clip_ratio = self.cfg.ppo.clip_ratio
        self.num_agents = cfg.general.num_workers
        self.num_epochs = 0
        self.epoch = 0
        self.steps_per_epoch = cfg.ppo.steps_per_epoch

        # multiprocessing
        self.manager = mp.Manager()
        self.shared_list = mp.Queue()
        self.timestep_counter = mp.Value('i', 0)
        self.lock = mp.Lock()
        self.rewards = mp.Queue()
        self.ep_lengths = mp.Queue()

        # create ac for training
        env, all_state_len = create_env(cfg)
        self.actor, self.critic = create_ac(cfg, all_state_len)

        # setup the distributed agents
        self.agents = [PPOAgent(cfg, env, all_state_len)
                       for _ in tqdm(range(self.num_agents))]

        self.cuda_device = cfg.general.cuda_device
        self.device = torch.device(f"cuda:{cfg.general.cuda_device}")
        self.actor.cuda(self.cuda_device)
        self.critic.cuda(self.cuda_device)

        self.configure_optimizers()
        self.mse = torch.nn.MSELoss()
        self.lam_r = self.cfg.env.lam_reward

        # logging
        now = datetime.now()
        #self.dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        self.d_string = now.strftime("%Y-%m-%d")

        try: 
            self.t_string
        except:
            self.t_string = now.strftime("%H-%M-%S")
        logdir = f'outputs/{self.d_string}/{self.t_string}'
        os.mkdir(logdir)
        self.writer = SummaryWriter(log_dir=logdir)

        self.param_in = torch.ones(
            cfg.ppo.batch_size, len(self.actor.paramlist)).cuda(self.cuda_device).float()

        # synchronize agents
        self.update_agents()

        # preallocate
        self.transitions = self.preallocate_memory()

        # handle data gen comparison
        self.param_path = cfg.env.param_path
        self.data_gen = cfg.env.param_data_gen_file
        self.keep_data_gen_loggin = False

        if self.data_gen is not None:
            self.keep_data_gen_loggin = True
            with open(f'{self.param_path}/{self.data_gen}', "r") as stream:
                self.params_gen = yaml.safe_load(stream)
            self.params_gen = np.array([float(x) for x in flatten_dict(self.params_gen)])

    def preallocate_memory(self, mydtype=np.float32):
        """for faster dataset creation"""
        dummy_transition = self.agents[0].return_single_state_transition()

        # Compute shapes for the preallocated arrays
        shapes = {
            attr: (self.steps_per_epoch,) +
            getattr(dummy_transition, attr).shape
            for attr in ('state', 'action', 'logp_old', 'qval', 'adv')
        }

        # Preallocate arrays with the computed shapes and specified data type
        preallocated_data = {
            attr: np.zeros(shape, dtype=mydtype)
            for attr, shape in shapes.items()
        }

        # Create and return a TransitionClass instance with preallocated arrays
        return TransitionClass(**preallocated_data)

    def setup_dis_processes(self):
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
                ),
            )
            p.start()
            self.processes.append(p)

    def configure_optimizers(self):
        """ Initialize Adam optimizer"""
        self.optimizer_actor = optim.Adam(
            self.actor.parameters(), lr=self.cfg.actor.lr)
        self.optimizer_critic = optim.Adam(
            self.critic.parameters(), lr=self.cfg.critic.lr)

    @timeit
    def collect_data(self):
        """distributed data collection using distributed proccesses"""

        # setup
        self.setup_dis_processes()
        self.collect_data_from_queue(self.shared_list)

        # wait for completion
        for p in self.processes:
            p.join()

        self.timestep_counter = mp.Value('i', 0)

        # recreate the dataloader with the new collected data
        self.dataloader = TransitionBatchDataLoader(
            CollectedDataDataset(self.transitions),
            batch_size=self.cfg.ppo.batch_size,
            shuffle=True,
            num_workers=self.cfg.general.num_trainers,
            cuda_device=self.cuda_device
        )
        self.download_ep_info_from_queue()

    def collect_data_from_queue(self, shared_queue: mp.Queue):
        """constantly unload the queue and add to dataset"""
        counter = 0

        # Store references to the arrays in variables to minimize array indexing
        state_arr = self.transitions.state
        action_arr = self.transitions.action
        logp_old_arr = self.transitions.logp_old
        qval_arr = self.transitions.qval
        adv_arr = self.transitions.adv

        # Refactor loop structure to avoid nested loops
        while counter < self.steps_per_epoch:
            try:
                data = shared_queue.get(block=True, timeout=5)
                state_arr[counter, :] = data.state
                action_arr[counter, :] = data.action
                logp_old_arr[counter, :] = data.logp_old
                qval_arr[counter] = data.qval
                adv_arr[counter] = data.adv
                counter += 1
            except:
                print('Queue empty')
                time.sleep(0.05)

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
        # factor slowly decreasing from 1 to 0
        epoch_fac = (self.cfg.general.epochs - self.epoch) / self.cfg.general.epochs
        for agent in self.agents:
            agent.update_ac(self.actor, self.critic, epoch_fac)

    @timeit
    def train_epoch(self):
        """training including a whole epoche"""
        self.total_loss_a = 0
        self.total_loss_c = 0
        self.num_batches = 0

        for _ in range(self.cfg.ppo.nb_optim_iters):
            self.train_full_dataset()

    def train_full_dataset(self):
        """go trough the whole dataset once"""

        for batch in self.dataloader:
            # train actor
            self.optimizer_actor.zero_grad()
            loss_a = self.actor_loss(batch)
            loss_a.backward()
            self.optimizer_actor.step()

            # train critic
            self.optimizer_critic.zero_grad()
            loss_c = self.critic_loss(batch)
            loss_c.backward()
            self.optimizer_critic.step()

            self.total_loss_a += loss_a.item()
            self.total_loss_c += loss_c.item()
            self.num_batches += 1

    def actor_loss(self, batch: TransitionBatch):
        """ppo loss for the actor"""
        # normalize advantages
        adv = (batch.adv - batch.adv.mean()) / batch.adv.std()
        adv = adv.float()

        logp_old = batch.logp_old[:, 0]

        batch_size = adv.shape[0]
        param_in = self.param_in[:batch_size]

        pi = self.actor.forward_fast(param_in)
        logp = self.actor.get_log_prob(pi, batch.action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio,
                               1 + self.clip_ratio) * adv.float()
        loss = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss

    def critic_loss(self, batch: TransitionBatch):
        """loss of the critic simply mse"""
        value = self.critic(batch.state)[:, 0]
        loss_critic = self.mse(batch.qval.float(), value)
        return loss_critic

    def run_training(self):
        """run the whole training"""
        for epoch in tqdm(range(self.cfg.general.epochs)):
            self.epoch = epoch
            self.collect_data()
            self.train_epoch()
            self.update_agents()
            self.log_progress()

    # functions for logging
    # ---------------------------------------------------------------------------------------------------
    @timeit
    def log_progress(self):
        avg_loss_a = self.total_loss_a / self.num_batches
        avg_loss_c = self.total_loss_c / self.num_batches

        self.writer.add_scalar(
            "ALLG/avg_ep_len", float(self.avg_ep_lengths), self.epoch)
        self.writer.add_scalar(
            "ALLG/avg_ep_loss", float((1 - self.avg_rewards) / self.lam_r), self.epoch)
        self.writer.add_scalar(
            "ALLG/loss_actor", float(avg_loss_a), self.epoch)
        self.writer.add_scalar(
            "ALLG/loss_critic", float(avg_loss_c), self.epoch)

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
            cur_flatten = np.array([float(x) for x in flatten_dict(parameters)])
            mse = np.mean((cur_flatten - self.params_gen)**2)
            self.writer.add_scalar('MSE-Params', float(mse), self.epoch)

    def log_figure(self):
        """log current progress in image"""
        if self.epoch % self.cfg.log.figure_intervall:
            return
        fig, err = self.agents[0].log_image_progress()
        fig.savefig(
            f'outputs/{self.d_string}/{self.t_string}/fig_{self.epoch}.png')
        self.writer.add_scalar(
            "ALLG/val_err", float(err), self.epoch)

    def save(self):
        """save the state of the trained model"""
        if self.epoch % self.cfg.log.param_save_intervall:
            return

        self.actor.eval()
        _, _, _, parameters = self.actor(self.param_in, sample=False)
        parameters = parameters[0]

        with open(f'outputs/{self.d_string}/{self.t_string}/parameters_{self.epoch}.yaml', 'w') as f:
            yaml.dump(parameters, f)
    # ---------------------------------------------------------------------------------------------------

# %%


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: OverallSettings):
    trainer = PPOSchedule(cfg)
    trainer.run_training()


# %%
if __name__ == '__main__':
    main()

# %%
