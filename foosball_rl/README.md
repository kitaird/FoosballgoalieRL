## Run Configuration
The run configuration file `foosball_rl/run_config.yml` contains the following sections:
- `Experiment_name`: The experiment name, which is used to store the results in the `experiments` directory.
- `Execution_Mode`: The execution mode `train` (= training new agents) or `eval` (= evaluating existing agents).
- `Env_id`: The environment-id to use, either `Goalkeeper-v0` or `Foosball-v0`.
- `Algorithm`: The algorithm to use for training. All algorithms from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and [stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) are supported.