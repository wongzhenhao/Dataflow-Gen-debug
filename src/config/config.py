from typing import Dict
from argparse import ArgumentError
from jsonargparse import ActionConfigFile, ArgumentParser

def init_config(args=None):
    """Initialize new configuration with updated settings."""
    parser = ArgumentParser(default_env=True, default_config_files=None)
    parser.add_argument('--config', action=ActionConfigFile, help='Path to a base config file', required=True)
    parser.add_argument('--meta_path', type=str, help='Meta path')
    parser.add_argument('--meta_folder', type=str, default=None, help='Image folder')
    parser.add_argument('--base_folder', type=str, default=None, help='Base folder')
    parser.add_argument('--save_folder', type=str, help='Save directory')
    parser.add_argument('--text_key', type=str, default=None, help='Text key')
    parser.add_argument('--image_key', type=str, default=None, help='Image key')
    parser.add_argument('--video_key', type=str, default=None, help='Video key')
    parser.add_argument('--steps', type=list, default=None, help='Model configurations')

    try:
        cfg = parser.parse_args(args=args)
        return cfg
    except ArgumentError:
        print('Configuration initialization failed')
