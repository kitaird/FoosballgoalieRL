## Execution mode Configuration
The execution mode configuration file `foosball_rl/modes/execution_mode_config.yml` contains the following sections:
- `Training`: Defines parameters for the training process:
  - `seeds`: The random seeds to use for the training process. The number of seeds defines the number of training runs.
  - `n_envs`: The number of parallel environments to use per training run. E.g if `seeds=[100, 200, 300]` and `n_envs=4`, the training process will run 3 training runs with 4 parallel environments each, assigning the seeds `[100, 101, 102, 103]` to the first run, `[200, 201, 202, 203]` to the second run, and `[300, 301, 302, 303]` to the third run. 
  - `total_timesteps`: The total number of timesteps to train the agent per training run. E.g. if `n_envs=4` and `total_timesteps=1000`, the agent will be trained for 1000 timesteps in total, with 250 timesteps per environment.
  - `tb_log_name`: The name of the tensorboard log.
  - `vec_normalize_load_path`: The path to load a potential vec_normalize path (e.g. in the case of resuming training).
<!-- -->
- `Evaluation`: Defines parameters for the evaluation process:
  - `eval_seeds`: The random seed to use for the evaluation process. Only one seed is supported for evaluation.
  - `n_eval_envs`: The number of parallel environments to use for evaluation. E.g. if `n_eval_envs=4`, the evaluation process will run 4 parallel environments.
  - `model_path`: The path to the model to load.
  - `vec_normalize_load_path`: The path to load a potential vec_normalize path (e.g. in the case of resuming training).
  - `n_eval_episodes`: The number of episodes to evaluate. The episodes are split among the parallel environments.
  