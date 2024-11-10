from typing import Dict
from argparse import ArgumentError
from jsonargparse import ActionConfigFile, ArgumentParser

def init_config(args=None):
    """Initialize new configuration with updated settings."""
    parser = ArgumentParser(default_env=True, default_config_files=None)
    parser.add_argument('--config', action=ActionConfigFile, help='Path to a base config file', required=True)
    parser.add_argument('--save_per_batch', type=bool, default=True, help='Save per batch')
    parser.add_argument('--meta_path', type=str, help='Meta path')
    parser.add_argument('--meta_folder', type=str, default=None, help='Image folder')
    parser.add_argument('--save_folder', type=str, help='Save directory')
    parser.add_argument('--data', type=Dict, help='Data configurations')
    parser.add_argument('--ImageCaptioner', type=Dict, default=None, help='Image Captioner configurations')
    parser.add_argument('--ImageGenerator', type=Dict, default=None, help='Image Generator configurations')
    parser.add_argument('--VideoCaptioner', type=Dict, default=None, help='Video Captioner configurations')
    parser.add_argument('--VideoGenerator', type=Dict, default=None, help='Video Generator configurations')

    try:
        cfg = parser.parse_args(args=args)
        return cfg
    except ArgumentError:
        print('Configuration initialization failed')
