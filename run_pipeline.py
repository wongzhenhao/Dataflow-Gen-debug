# run_pipeline.py
import os
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
    datefmt='%m-%d %H:%M:%S'
)

from src.config import init_config
from src.pipeline.manager import PipelineManager

def run(config: dict, base_save_folder: str, final_output_folder: str):
    """
    Initialize and run the pipeline manager with the given configuration and folders.

    :param config: Configuration dictionary
    :param base_save_folder: Path to save intermediate results
    :param final_output_folder: Path to save final results
    """
    manager = PipelineManager(
        config=config,
        base_save_folder=base_save_folder,
        final_output_folder=final_output_folder
    )
    manager.run()

def main():
    """
    Main function to initialize configuration and run the pipeline.
    """
    config = init_config()
    base_save_folder = config.get('base_folder', 'intermediate_results/')
    final_output_folder = config.save_folder
    
    run(config, base_save_folder, final_output_folder)

if __name__ == '__main__':
    main()