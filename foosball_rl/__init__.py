from pathlib import Path

import yaml

run_config_path = Path(__file__).parent / 'run_config.yml'
with open(run_config_path) as f:
    run_config = yaml.safe_load(f)

EXPERIMENT_NAME = run_config['Experiment_name']
EXECUTION_MODE = run_config['Execution_mode']
ENV_ID = run_config['Env_id']
RL_ALGORITHM = run_config['Algorithm']