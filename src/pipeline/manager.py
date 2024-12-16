# src/pipeline/manager.py

import logging
import os
from typing import Any, List, Dict
from src.pipeline.steps import PipelineStep, PipelinePreprocessStep, PipelinePostprocessStep
from src.pipeline.wrappers import (
    ImageCaptionerWrapper, ImageGeneratorWrapper,
    VideoCaptionerWrapper, VideoGeneratorWrapper,
    TextGeneratorWrapper
)
from src.data.DataManager import DataManager

class PipelineManager:
    """
    Manages the execution of the pipeline steps based on the provided configuration.
    """
    STEP_MAPPING = {
        'preprocess': PipelinePreprocessStep,
        'postprocess': PipelinePostprocessStep,
        'ImageCaptioner': ImageCaptionerWrapper,
        'ImageGenerator': ImageGeneratorWrapper,
        'VideoCaptioner': VideoCaptionerWrapper,
        'VideoGenerator': VideoGeneratorWrapper,
        'TextGenerator': TextGeneratorWrapper,
    }

    def __init__(self, config: dict, base_save_folder: str, final_output_folder: str):
        """
        Initialize the PipelineManager with configuration and folders.

        :param config: Configuration dictionary
        :param base_save_folder: Path to save intermediate results
        :param final_output_folder: Path to save final results
        """
        self.steps_config = config.get('steps', [])
        preprocess_step = {
            'type': 'preprocess',
            'name': 'format',
            'config': {
                'text_key': config.get('text_key', None),
                'image_key': config.get('image_key', None),
                'video_key': config.get('video_key', None),
                'meta_folder': config.get('meta_folder', None)
            }
        }
        postprocess_step = {
            'type': 'postprocess',
            'name': 'format',
            'config': {
                'text_key': config.get('text_key', None),
                'image_key': config.get('image_key', None),
                'video_key': config.get('video_key', None),
                'meta_folder': config.get('meta_folder', None)
            }
        }
        self.steps_config = [preprocess_step] + self.steps_config + [postprocess_step]
        
        self.base_save_folder = base_save_folder
        self.final_output_folder = final_output_folder
        self.data_manager = DataManager(base_save_folder=self.base_save_folder, final_output_folder=self.final_output_folder)
        self.current_input = config.get('meta_path')  # Initial input

    def run(self):
        """
        Execute all pipeline steps in sequence.
        """
        for idx, step_conf in enumerate(self.steps_config):
            step_type = step_conf['type']
            step_name = step_conf['name']
            step_config = step_conf['config']
            step_save_name = f"step_{idx}_{step_type}_{step_name}"

            logging.info(f"Starting step {idx}: {step_type} - {step_name}")

            step_class = self.STEP_MAPPING.get(step_type)
            if not step_class:
                logging.error(f"Unknown step type: {step_type}")
                continue

            step_instance = step_class(
                name=step_save_name,
                config=step_config,
                data_manager=self.data_manager
            )

            try:
                self.current_input = step_instance.execute(self.current_input)
                logging.info(f"Completed step {idx}: {step_type} - {step_name}")
            except Exception as e:
                logging.error(f"Error in step {step_name}: {e}", exc_info=True)
                raise e
        
        logging.info("Pipeline execution completed.")