from pathlib import Path

import yaml

config_path = Path(__file__).parent.parent / 'run_config.yml'


def get_run_config():
    if not hasattr(get_run_config, 'config'):
        with open(config_path) as f:
            get_run_config.config = yaml.safe_load(f)
    return get_run_config.config

