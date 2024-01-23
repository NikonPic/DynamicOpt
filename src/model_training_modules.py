# %%
from collections import deque, namedtuple
import random
import numpy as np
from torch.utils.data.dataset import IterableDataset
from model_simulation import MeasureDataset, KneeModel
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Normal
import yaml
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass

PARAM_PATH = '../params/parameters_init.yaml'

@dataclass
class TransitionClass:
    """Dataclass containing all relevant states"""
    state: torch.tensor
    measure_state: torch.tensor
    ft: torch.tensor
    ft_ext_cur: torch.tensor
    params: torch.tensor
    pos: torch.tensor
    concat_state: torch.tensor
    next_state: torch.tensor
    measure_next_state: torch.tensor
    tpos: torch.tensor


TransitionTuple = namedtuple(
    "TransitionTuple",
    field_names=list(TransitionClass.__dataclass_fields__),
)


class TransitionBatch:
    """Class to handle batch of states"""

    def __init__(self, translist: list[TransitionClass]):
        self.translist = translist
        self.len = len(translist)
        self.fill_class_fields()

    def fill_class_fields(self):
        """batch with same fields as transition"""
        for fieldname in TransitionClass.__dataclass_fields__:
            field_result = torch.stack(
                [getattr(tran, fieldname) for tran in self.translist], dim=0).float()
            setattr(self, fieldname, field_result)


class ReplayMemory(object):
    """Stores all transisitons"""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition: TransitionClass):
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReplayDataset(IterableDataset):
    """Dataset for pytorch dataloaders with updating dataset"""

    def __init__(self, memory: ReplayMemory, batch_size: int = 256) -> None:
        self.memory = memory
        self.batch_size = batch_size

    def __iter__(self):
        batch = TransitionBatch(self.memory.sample(self.batch_size))

        for i in range(batch.len):
            yield self.get_local_tuple(batch, i)

    def get_local_tuple(self, batch: TransitionBatch, idx: int):
        my_tuple = []
        for fieldname in TransitionClass.__dataclass_fields__:
            myval = getattr(batch, fieldname)[idx]
            my_tuple.append(myval)
        return TransitionTuple(*tuple(my_tuple))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SACDeepCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256), activation=nn.LeakyReLU, tau=0.005, dropout_rate=0.05):
        super(SACDeepCritic, self).__init__()
        self.dropout_rate = dropout_rate

        self.q1 = self._build_q_net(
            state_dim, action_dim, hidden_dims, activation)
        self.q2 = self._build_q_net(
            state_dim, action_dim, hidden_dims, activation)

        self.q1_target = self._build_q_net(
            state_dim, action_dim, hidden_dims, activation)
        self.q2_target = self._build_q_net(
            state_dim, action_dim, hidden_dims, activation)

        # Initialize target networks with the same weights as the original networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.tau = tau

    def _build_q_net(self, state_dim, action_dim, hidden_dims, activation, use_dropout=True, output_dim=1):
        layers = []
        input_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(input_dim, hidden_dim),
                LayerNorm(hidden_dim),
                activation(),
            ]
            input_dim = hidden_dim

        # Add dropout only if use_dropout flag is set
        if use_dropout:
            layers.append(nn.Dropout(p=self.dropout_rate))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def q1_forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)

    def q2_forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q2(x)

    def q1_target_forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_target(x)

    def q2_target_forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q2_target(x)

    def forward(self, state, action):
        q1 = self.q1_forward(state, action)
        q2 = self.q2_forward(state, action)
        return q1, q2

    def update_target_networks(self):
        with torch.no_grad():
            for q1_params, q1_target_params, q2_params, q2_target_params in zip(self.q1.parameters(), self.q1_target.parameters(), self.q2.parameters(), self.q2_target.parameters()):
                q1_target_params.data.copy_(
                    self.tau * q1_params.data + (1 - self.tau) * q1_target_params.data)
                q2_target_params.data.copy_(
                    self.tau * q2_params.data + (1 - self.tau) * q2_target_params.data)


def flatten_dict(dct):
    def flatten(item):
        if isinstance(item, dict):
            for _, value in item.items():
                yield from flatten(value)
        elif isinstance(item, list):
            for value in item:
                yield from flatten(value)
        else:
            yield item

    return list(flatten(dct))


class ParamLearner(torch.nn.Module):
    """Class to conbtain the NN for Parameter Learning"""

    def __init__(self, width=128, use_std=False, reinforce=False, param_file=None) -> None:
        super(ParamLearner, self).__init__()
        self.param_file = param_file if param_file is not None else PARAM_PATH
        self.load_parameters()
        self.paramlist, self.param_names = self.params2list(
            [], [], self.params)

        self.eps = 1e-6
        self.epoch_fac = 1
        self.paramlen = len(self.paramlist)
        self.out_len = self.paramlen * 2 if use_std else self.paramlen
        self.construct_network(width)

        self.get_param_scale()

    def construct_network(self, width):
        layers = [
            nn.BatchNorm1d(self.paramlen),
            nn.Linear(self.paramlen, width), nn.Softplus(),
            nn.Linear(width, width), nn.Softplus(),
            nn.Linear(width, width), nn.Softplus(),
            nn.Linear(width, self.out_len)
        ]
        self.network = nn.Sequential(*layers)
        self.extra_parameters = Parameter(
            torch.randn(len(self.paramlist)) * 0.05)
        self.extra_scale = torch.tensor(
            [0 if p > 0 else 1 for p in self.paramlist])
        self.std_replace = torch.ones(self.paramlen) * 0.02

        # log std for reinforcement learning
        log_std = -3 * torch.ones(self.paramlen, dtype=torch.float)
        self.log_std = torch.nn.Parameter(log_std)

    def get_param_scale(self):
        self.param_in = torch.ones(5, len(self.paramlist))
        self.paramscale = []
        self.paramoffset = []
        self.net_offset = []

        net_out = self.forward_fast(self.param_in).loc[0, :].detach().numpy()

        for (para, net_val_init) in zip(self.paramlist, net_out):
            loc_s = 0.5 * para
            loc_off = para
            net_off = net_val_init

            self.paramscale.append(loc_s)
            self.paramoffset.append(loc_off)
            self.net_offset.append(net_off)

        self.paramscale = torch.tensor(self.paramscale)
        self.paramoffset = torch.tensor(self.paramoffset)
        self.net_offset = torch.tensor(self.net_offset)

        # later will be
        # new_para = loc_scale * (net_out-net_out_init) + loc_offset

    def load_parameters(self):
        """load the required parameters from yaml"""
        with open(self.param_file, "r") as stream:
            try:
                self.params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit(0)

    def params2list(self, mylist: list, mynames: list, parameters: dict):
        """generate a list from all parameters"""
        for key in parameters:
            if (isinstance(parameters[key], dict)):
                mylist, mynames = self.params2list(
                    mylist, mynames, parameters[key])
            else:
                split_params = parameters[key].split(' ')
                for split_param in split_params:
                    mylist.append(float(split_param))
                mynames.append(key)

        return mylist, mynames

    def list2params(self, list_count: int, mylist: list, parameters: dict):
        """refill the params with the list"""
        for key in parameters:
            if (isinstance(parameters[key], dict)):
                list_count, parameters[key] = self.list2params(
                    list_count, mylist, parameters[key])
            else:
                split_params = parameters[key].split(' ')
                loc_str = ''
                for _ in split_params:
                    loc_str = f'{loc_str} {mylist[list_count]}'
                    list_count += 1
                loc_str = loc_str[1:]
                parameters[key] = loc_str
        return list_count, parameters

    def forward(self, x, sample=True, reparametrize=True):
        mu = self.network(x)
        mu = torch.tanh(mu)

        std = torch.exp(self.log_std.type_as(x)) + self.eps
        pi = Normal(loc=mu, scale=std)

        if sample:
            if reparametrize:
                action = pi.rsample()
            else:

                noise = torch.randn_like(mu)
                action = mu + noise * std
        else:
            action = mu

        out_params = (action - self.net_offset.type_as(x)) * \
            self.paramscale.type_as(x) + self.paramoffset.type_as(x)
        out_params_dicts = self.rewrite_parameters(out_params)
        log_prob = self.get_log_prob(pi, action)

        return pi, action, log_prob, out_params_dicts

    def forward_fast(self, x):
        """faster form for training only"""
        mu = self.network(x)
        mu = torch.tanh(mu)

        std = torch.exp(self.log_std.type_as(x)) + self.eps
        pi = Normal(loc=mu, scale=std)
        return pi

    def get_act_logp(self, x):
        pi = self.forward_fast(x)
        action = pi.rsample()
        log_p = self.get_log_prob(pi, action)
        return action, log_p

    def get_log_prob_for_actions(self, x, actions: torch.Tensor):
        pi = self.forward_fast(x)
        log_p = self.get_log_prob(pi, actions)
        return log_p

    def get_log_prob(self, pi, actions: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(actions).sum(axis=-1)

    def rewrite_parameters(self, out_params):
        """write the local parameters for the batch for simulator"""
        out_lists = out_params.detach().cpu().numpy()
        out_lists = [list(out_lists[i, :]) for i in range(out_lists.shape[0])]

        out_params_list = []
        for out_list in out_lists:
            loc_parameters = copy.deepcopy(self.params)
            _, out_params = self.list2params(0, out_list, loc_parameters)
            out_params_list.append(out_params)

        return out_params_list


class Critic(nn.Module):
    def __init__(self, input_len, output_len, width=128):
        super(Critic, self).__init__()
        self.input_len = input_len
        self.output_len = output_len

        self.network = nn.Sequential(
            nn.BatchNorm1d(self.input_len),
            nn.Linear(self.input_len, width),
            nn.BatchNorm1d(width),
            nn.Softplus(),
            nn.Linear(width, 2*width),
            nn.BatchNorm1d(2*width),
            nn.Softplus(),
            nn.Linear(2*width, 2*width),
            nn.BatchNorm1d(2*width),
            nn.Softplus(),
            nn.Linear(2*width, width),
            nn.BatchNorm1d(width),
            nn.Softplus(),
            nn.Linear(width, output_len),
        )

    def forward(self, x):
        return self.network(x)


class SimLearner(nn.Module):
    """at each timestep -> predict next timestep"""

    def __init__(self, param_len, obs_len=7, ft_len=6, width=128, st_len=7, dst_len=6) -> None:
        super(SimLearner, self).__init__()
        self.input_len = param_len + obs_len + ft_len
        self.network = nn.Sequential(
            nn.Linear(self.input_len, width),
            nn.BatchNorm1d(width),
            nn.Softplus(),
            nn.Linear(width, 2*width),
            nn.BatchNorm1d(2*width),
            nn.Softplus(),
            nn.Linear(2*width, 2*width),
            nn.BatchNorm1d(2*width),
            nn.Softplus(),
            nn.Linear(2*width, 2*width),
            nn.BatchNorm1d(2*width),
            nn.Softplus(),
            nn.Linear(2*width, width),
            nn.BatchNorm1d(width),
            nn.Softplus(),
            nn.Linear(width, width),
        )
        self.head_state = nn.Sequential(
            nn.Linear(width, st_len),
        )
        self.head_dstate = nn.Sequential(
            nn.Linear(width, dst_len),
        )

    def forward(self, x):
        """predict next state"""
        latent = self.network(x)
        state = self.head_state(latent)
        dstate = self.head_dstate(latent)
        if len(dstate.shape) > 1:
            all_state = torch.concat((state, dstate), dim=1)
        else:
            all_state = torch.concat((state, dstate), dim=0)
        return state, dstate, all_state


class SimHelper(nn.Module):
    """
    at each timestep: provide a correction ft input
    that will enable the simulation to stay close to measurement
    """

    def __init__(self, param_len, state_len=7, ft_len=6, obs_len=6, width=128) -> None:
        super(SimHelper, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.input_len = param_len + state_len + ft_len + obs_len
        self.network = nn.Sequential(
            nn.Linear(self.input_len, width),
            nn.BatchNorm1d(width),
            nn.Softplus(),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.Softplus(),
            nn.Linear(width, 2*width),
            nn.BatchNorm1d(2*width),
            nn.Softplus(),
            nn.Linear(2*width, 2*width),
            nn.BatchNorm1d(2*width),
            nn.Softplus(),
            nn.Linear(2*width, width),
            nn.BatchNorm1d(width),
            nn.Softplus(),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.Softplus(),
            nn.Linear(width, ft_len),
            nn.Tanh()
        )
        self.force_scale = 30
        self.torque_scale = 5
        self.scale = torch.tensor([
            self.force_scale,
            self.force_scale,
            self.force_scale,
            self.torque_scale,
            self.torque_scale,
            self.torque_scale,
        ])

    def forward(self, x):
        """apply correction ft"""
        out_network = self.network(x) * self.scale.type_as(x)
        return out_network


class Agent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self, env, memory: ReplayMemory, optimizte_direct: bool = False,
                 seg_len=25, ft_len=6, obs_len=7, paramlen=39, statelen=25, freq=100,
                 device='cpu') -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.optimize_direct = optimizte_direct
        self.env = env
        self.ft_len = ft_len
        self.obs_len = obs_len
        self.paramlen = paramlen
        self.statelen = statelen
        self.seg_len = seg_len
        self.memory = memory
        self.freq = freq
        self.dset = MeasureDataset(mode='train', seg_len=seg_len, shuffle=True)
        self.iterset = iter(self.dset)
        self.seg_data = next(self.iterset)
        self.epoch_count = 0
        self.episode_count = 0
        self.tpos = 0
        self.ft_corr_scale = 0 if self.optimize_direct else 1
        self.param_in = torch.ones(1, paramlen).to(device)
        self.qpos = np.zeros((seg_len, obs_len))

        # predefined batches
        self.siml_in_len = self.paramlen + self.ft_len + self.statelen
        self.simh_in_len = self.paramlen + self.ft_len + self.statelen + self.obs_len

    def new_epoche(self):
        self.dset.reshuffle()
        self.iterset = iter(self.dset)
        self.episode_count = 0

    @torch.no_grad()
    def reset(self, paraml=nn.Module, device='cpu') -> None:
        """Resents the environment and updates the state."""
        # store old infos
        self.last_episode_len = self.tpos
        self.last_trajecory = self.qpos.copy()
        self.lastsegment = copy.deepcopy(self.seg_data)

        # reset time position
        self.tpos = 0

        # get new trajectory
        self.seg_data = next(self.iterset)
        self.episode_count += 1

        if self.episode_count >= self.dset.len - 1:
            self.new_epoche()
            self.epoch_count += 1

        # sample parameters
        paraml.eval()
        self.param_in = torch.ones(1, self.paramlen).to(device)
        self.p, self.q, self.normed_params, parameters = paraml(self.param_in)
        self.parameters = parameters[0]

        # set init state, ft_ext curve
        self.tend, self.ft_ext = self.env.reset(self.seg_data, self.parameters)
        self.state = self.env.state()
        self.pos = self.env.pos()

        self.qpos = np.zeros((self.seg_len, self.obs_len))
        self.qpos[self.tpos, :] = self.pos

    def get_measure_state(self, device):
        pos = torch.tensor(self.seg_data['tp'][self.tpos, :])
        quat = torch.tensor(self.seg_data['tq'][self.tpos, :])
        s_t_mea = torch.zeros(7, dtype=torch.float64)
        s_t_mea[:3] = pos
        s_t_mea[3:] = quat
        return s_t_mea.to(device)

    def get_complete_state_simh(self, ft_ext_cur, s_t, s_mea_t, device):
        """fill the input for the helper network"""
        simh_in = torch.zeros(1, self.simh_in_len).to(device)
        simh_in[:, :self.paramlen] = self.normed_params
        simh_in[:, self.paramlen:self.paramlen+self.ft_len] = ft_ext_cur
        simh_in[:, self.paramlen+self.ft_len:self.paramlen +
                self.ft_len + self.statelen] = s_t
        simh_in[:, self.paramlen+self.ft_len + self.statelen:] = s_mea_t
        return simh_in

    def get_complete_state_siml(self, ft, s_t, device):
        """fill the input for the learner network"""
        siml_in = torch.zeros(1, self.siml_in_len).to(device)
        siml_in[:, :self.paramlen] = self.normed_params
        siml_in[:, self.paramlen:self.paramlen+self.ft_len] = ft
        siml_in[:, self.paramlen+self.ft_len:] = s_t
        return siml_in

    @torch.no_grad()
    def play_step(self, paraml: nn.Module, simh: nn.Module, device: str = "cpu"):
        """perform a single step on the enviornment"""
        # current states
        ft_ext_cur = torch.tensor(self.ft_ext[self.tpos, :]).to(device)
        s_t = torch.tensor(self.env.state()).to(device)
        pos_t = torch.tensor(self.env.pos()).to(device)
        s_mea_t = self.get_measure_state(device)
        simh_in = self.get_complete_state_simh(
            ft_ext_cur, s_t, s_mea_t, device)

        # apply the sim helper
        simh.eval()
        ft_corr = simh(simh_in) * self.ft_corr_scale

        # the input for the simulator
        ft_sim = ft_ext_cur.to(device) + ft_corr[0, :]
        ft_sim_in = list(ft_sim.cpu().numpy())

        # the concat state
        s_cat_t = self.get_complete_state_siml(ft_sim, s_t, device)

        # interact with enviornment
        new_state, _, done = self.env.step(ft_sim_in)

        # prepare new state
        self.tpos += 1
        s_t1 = torch.tensor(new_state).to(device)

        try:
            s_mea_t1 = self.get_measure_state(device)
        except IndexError:
            print(self.tpos)
            print(done)
            s_mea_t1 = s_mea_t
            done = True

        # create new transition
        local_transition = TransitionClass(
            state=s_t.cpu(),
            measure_state=s_mea_t.cpu(),
            ft=ft_sim.cpu(),
            ft_ext_cur=ft_ext_cur.cpu(),
            params=self.normed_params[0].cpu(),
            pos=pos_t.cpu(),
            concat_state=s_cat_t[0, :].cpu(),
            next_state=s_t1.cpu(),
            measure_next_state=s_mea_t1.cpu(),
            tpos=torch.tensor(self.tpos).cpu(),
        )
        self.memory.push(local_transition)

        if done:
            self.reset(paraml, device=device)

        return done

    @torch.no_grad()
    def run_episode(self, paraml: nn.Module, simh: nn.Module, device: str = "cpu"):
        """play a whole episode at once"""
        done = False
        while not done:
            done = self.play_step(paraml, simh, device)

        return self.last_episode_len

    def log_image_progress(self, paraml: ParamLearner):
        """save an image to the session"""
        seg_data = self.lastsegment
        qpos = self.last_trajecory

        len_vec = len(seg_data['fp'][:, 0])
        tend = len_vec / self.freq
        time_vec = np.linspace(0, tend, len_vec)
        fig1 = plt.figure(figsize=(12, 12))

        plt.subplot(2, 2, 1)
        plt.plot(time_vec, seg_data['fp'][:len_vec, :])
        plt.ylabel('true pos in m')
        plt.ylim([-1, 1])
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(time_vec, qpos[:, :3])
        plt.ylabel('sim pos in m')
        plt.ylim([-1, 1])
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(time_vec, seg_data['fq'][:len_vec, :])
        plt.ylabel('true qaut')
        plt.xlabel('time in s')
        plt.ylim([-1, 1])
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.plot(time_vec, qpos[:, 3:])
        plt.ylabel('sim qaut')
        plt.xlabel('time in s')
        plt.ylim([-1, 1])
        plt.grid()

        # get the validation as well
        _, _, _, parameters = paraml(
            self.param_in, sample=False)
        parameters = parameters[0]

        fig2 = self.env.perform_all(parameters)

        return fig1, fig2




# %% Initialize
if __name__ == '__main__':
    pass