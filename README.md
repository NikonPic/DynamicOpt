# Knee Model (MJCF)

## Overview

This package contains a simplified robot description (MJCF) of the human knee model. 

<p float="middle">
  <img src="img/knee.png" width="400">
</p>




## Class Structure

```mermaid
---
title:  Knee Simulation, Reinforcement Learning, and Neural Network Classes
---

classDiagram
    MeasureDataset -- KneeModel
    ParameterHandler <|-- ModelRenderer
    ModelRenderer <|-- KneeModel
    TransitionClass -- TransitionTuple
    TransitionClass -- TransitionBatch
    TransitionBatch -- CollectedDataDataset
    CollectedDataDataset -- TransitionBatchDataLoader
    TransitionBatchDataLoader --> TransitionBatch
    RLAgent -- PPOSchedule
    PPOSchedule *-- ParamLearner
    RLAgent *-- ParamLearner
    Critic *-- ParamLearner
    PPOSchedule -- TransitionBatchDataLoader
    RLAgent -- KneeModel: env
    PPOSchedule *-- Critic

    class MeasureDataset {
        mode : str
        train_per : float
        seg_len : int
        shuffle : bool
        use_fake : bool
        data_file : str
        measure_path : str
        lam_reward : float
        data : pd.DataFrame
        offset_pos : int
        len_data : int
        len : int
        index_list : list
        model : KneeModel
    }

    class ParameterHandler {
        parameters_file : str
        parameters : dict
    }

    class ModelRenderer {
        template_file : str
        path_ext : str
        raw_model_str : str
        model : mujoco_py.MjModel
        physics : mujoco_py.MjSim
    }

    class KneeModel {
        crit_speed : float
        freq : float
        dt : float
        done : bool
        data : pd.DataFrame
        seg_data : np.ndarray
        t_step : int
        mse : float
        lam_reward : float
        epoch_count : int
        episode_count : int
        ft_ext : np.ndarray
        dset_active : MeasureDataset
        tend : float
        max_sim_len : int
    }

    class TransitionClass {
        state : torch.tensor
        action : torch.tensor
        logp_old : torch.tensor
        qval : float
        adv : float
    }
    
    class TransitionTuple {
    }
    
    class TransitionBatch {
        translist : list
        len : int
        cuda_device : int
    }
    
    class CollectedDataDataset {
        transitions : list
    }
    
    class TransitionBatchDataLoader {
        dataset : Dataset
        _batch_size : int
        _shuffle : bool
        cuda_device : int
    }
    
    class RLAgent {
        gamma : float
        lam : float
        env : KneeModel
        all_state_len : int
        actor : ActorCritic
        critic : ActorCritic
        device : string
        steps_per_epoch : int
        param_in : torch.tensor
    }
    
    class PPOSchedule {
        cfg : OverallSettings
        clip_ratio : float
        num_agents : int
        num_epochs : int
        steps_per_epoche : int
        shared_queue : Queue
        timestep_counter : Value
        agents : list[RLAgent]
        processes : list[Process]
        actor : ActorCritic
        critic : ActorCritic
        cuda_device : int
        writer : SummaryWriter
        param_in : torch.tensor
    }

    class ParamLearner {
        width : int
        use_std : bool
        reinforce : bool
        param_file : str
        params : dict
        paramlist : list
        param_names : list
        paramlen : int
        paramscale : torch.tensor
        paramoffset : torch.tensor
        extra_parameters
        extra_scale : torch.tensor
        log_std : torch.tensor
        network : nn.Sequential
    }
    
    class Critic {
        input_len : int
        output_len : int
        width : int
        network : nn.Sequential
    }
```

## License

This model is released under an [MIT License](LICENSE).