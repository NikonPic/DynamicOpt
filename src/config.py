# %% file to map the configs in yaml strucutre
from dataclasses import dataclass
from typing import List


@dataclass
class ActorSettings:
    lr: float
    width: int
    use_std: bool
    reinforce: bool
    max_grad_norm: float


@dataclass
class CriticSettings:
    lr: float
    width: float
    hidden_dims: List[int]


@dataclass
class EnvSettings:
    model_type: str
    path_ext: str
    scene_path: str
    param_path: str
    param_file: str
    param_data_gen_file: str
    crit_speed: float
    max_episode_len: int
    ft_input_len: int
    freq: int
    apply_ft_body_id: int
    lam_reward: float
    data_path: str


@dataclass
class GenSettings:
    num_workers: int
    num_trainers: int
    epochs: int
    cuda_device: int
    seed: int


@dataclass
class PPOSettings:
    gamma: float
    lam: float
    batch_size: int
    steps_per_epoch: int
    nb_optim_iters: int
    clip_ratio: float


@dataclass
class SACSettings:
    tau: float
    gamma: float
    learn_alpha: bool
    alpha_init: float
    alpha_lr: int
    batch_size: int
    steps_per_epoch: int
    nb_optim_iters: int
    memory_size: int
    nb_batch_ids: int
    cql_weight: float
    critic_dropout_rate: float

@dataclass
class SimpleSettings:
    batch_size: int
    clip_ratio: float
    steps_per_epoch: int
    nb_optim_iters: int
    nb_batch_ids: int
    memory_size: int


@dataclass
class MeasureSettings:
    train_percentage: int
    shuffle: bool
    measure_path: str
    filter_oscillations: bool
    filter_apps: List[int]
    filter_tste: List[int]
    use_start_only: bool
    start_pos: int


@dataclass
class LoggingSettings:
    param_intervall: int
    figure_intervall: int
    param_save_intervall: int


@dataclass
class OverallSettings:
    actor: ActorSettings
    critic: CriticSettings
    env: EnvSettings
    general: GenSettings
    ppo: PPOSettings
    measure_dataset: MeasureSettings
    log: LoggingSettings
    simple: SimpleSettings


# %%
