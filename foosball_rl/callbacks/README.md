## Callback Configuration
The callback configuration file `foosball_rl/callbacks/callback_config.yml` contains the following sections:
- `Callbacks`: Defines parameters for the callbacks during training/evaluation:
  - `use_tensorboard_callback`: Whether to use the Tensorboard logging.
  - `use_progress_bar_callback`: Whether to use the `ProgressBarCallback` to display a progress bar during training.
  - `use_eval_callback`: Whether to use the `EvalCallback` to evaluate the model during training.
  - `use_checkpoint_callback`: Whether to use the `CheckpointCallback` to save the model during training.
<!-- -->
- `EvalCallback`: Properties for the `EvalCallback`.
  - `n_eval_envs`: The number of parallel environments to use for evaluation. E.g. if `n_eval_envs=4`, the evaluation process will run 4 parallel environments.
  - `n_eval_episodes`: The number of episodes to evaluate. The episodes are split among the parallel environments.
  - `eval_seed`: The random seed to use for the evaluation process. Only one seed is supported for evaluation. When using multiple eval envs, the eval_seed will be incremented for each eval, so that all eval envs have different seeds.
  - `eval_freq`: The frequency of intermediate evaluations in timesteps.
  - `eval_deterministic`: Whether to use a deterministic policy for evaluation.
<!-- -->
- `CheckpointCallback`: Properties for the `CheckpointCallback`.
  - `name_prefix`: The prefix for the checkpoint files.
  - `checkpoint_save_freq`: The interval of saving the model in timesteps.
  - `checkpoint_save_replay_buffer`: Whether to save the replay buffer if existing.
  - `checkpoint_save_vecnormalize`: Whether to save the vec_normalize if existing.