from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
config_path = base_dir / 'resources' / 'config.ini'


def get_config():
    if not hasattr(get_config, 'config'):
        get_config.config = ConfigParser(interpolation=ExtendedInterpolation())
        get_config.config.read(config_path)
    return get_config.config


def save_run_info(config: dict[str, dict], save_path: Path, seed: int, algorithm_name: str):
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    LINE_SEPARATOR = '-----------------\n'
    with open(save_path / 'run_configuration.txt', 'w') as f:
        f.write('Run Configuration\n')
        f.write(LINE_SEPARATOR)
        f.write(f"Experiment name: {config['Common']['experiment_name']}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Algorithm: {algorithm_name}\n")
        f.write(LINE_SEPARATOR)
        f.write('Arguments\n')
        for k, v in config._sections.items():
            if isinstance(v, dict):
                f.write(LINE_SEPARATOR)
                f.write(f'{k}\n')
                for i in v.items():
                    f.write(f'\t{i}\n')
                f.write(LINE_SEPARATOR)
        f.write('\n')
        f.write('')
