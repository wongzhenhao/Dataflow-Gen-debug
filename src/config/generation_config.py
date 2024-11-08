from typing import Dict
from argparse import ArgumentError
from jsonargparse import ActionConfigFile, ArgumentParser

def init_generation_config(args=None):
    """Initialize new configuration with updated settings."""
    parser = ArgumentParser(default_env=True, default_config_files=None)
    parser.add_argument('--config', action=ActionConfigFile, help='Path to a base config file', required=True)
    parser.add_argument('--meta_prompt_path', type=str, help='Meta prompt path')
    parser.add_argument('--save_folder', type=str, help='Save directory')
    parser.add_argument('--data', type=Dict, help='Data configurations')
    parser.add_argument('--generators', type=Dict, required=True, help='Scorer configurations')

    try:
        cfg = parser.parse_args(args=args)
        return cfg
    except ArgumentError:
        print('Configuration initialization failed')
