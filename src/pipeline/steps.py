# src/pipeline/steps.py

import os
from abc import ABC, abstractmethod
from typing import Any, List, Dict

from src.data.DataManager import Recorder
from src.utils.data_utils import load_from_data_path

class PipelineStep(ABC):
    def __init__(self, name: str, config: dict):
        """
        Initialize the pipeline step with a name and configuration.

        :param name: Name of the step
        :param config: Configuration dictionary for the step
        """
        self.name = name
        self.config = config

    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """
        Execute the pipeline step.

        :param input_data: Data from the previous step
        :return: Data to be passed to the next step
        """
        pass


class PipelinePreprocessStep(PipelineStep):
    """
    Preprocess step to format data according to the specified keys.
    """
    def __init__(self, name: str, config: dict, data_manager):
        super().__init__(name, config)
        self.data_manager = data_manager

    def execute(self, input_data: str) -> str:
        """
        Load data, reformat it based on keys, and record the new data.

        :param input_data: Path to the metadata
        :return: Path to the saved preprocessed data
        """
        data = load_from_data_path(input_data)
        text_key = self.config.get('text_key')
        image_key = self.config.get('image_key')
        video_key = self.config.get('video_key')
        meta_folder = self.config.get('meta_folder')

        new_data = [
            self._reformat_item(item, text_key, image_key, video_key, meta_folder)
            for item in data
        ]

        recorder = Recorder(
            data_manager=self.data_manager,
            step_name=self.name,
        )
        recorder.record(new_data)
        return recorder.dump()
    
    @staticmethod
    def _reformat_item(item: Dict, text_key: str, image_key: str, video_key: str, meta_folder: str) -> Dict:
        """
        Reformat a single data item based on specified keys.

        :param item: Original data item
        :param text_key: Key for text data
        :param image_key: Key for image data
        :param video_key: Key for video data
        :param meta_folder: Folder path for metadata
        :return: Reformatted data item
        """
        new_item = {k: v for k, v in item.items() if k not in [text_key, image_key, video_key]}

        if text_key and text_key in item:
            new_item['raw_text'] = item[text_key]
            new_item['text'] = item[text_key]

        if image_key and image_key in item:
            new_item['raw_image'] = item[image_key]
            new_item['image'] = os.path.abspath(os.path.join(meta_folder, item[image_key]))

        if video_key and video_key in item:
            new_item['raw_video'] = item[video_key]
            new_item['video'] = os.path.abspath(os.path.join(meta_folder, item[video_key]))

        return new_item


class PipelinePostprocessStep(PipelineStep):
    """
    Postprocess step to reformat data after processing steps.
    """
    def __init__(self, name: str, config: dict, data_manager):
        super().__init__(name, config)
        self.data_manager = data_manager

    def execute(self, input_data: str) -> str:
        self.data_manager.save_final_results()
