# src/data/Dataset.py

import os
import json
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from typing import Any, List, Dict

from src.utils.data_utils import load_from_data_path

class DataFlowDataset(Dataset):
    """
    Abstract base class for all datasets in the pipeline.
    """
    def __init__(self, meta_path: str, save_folder: str):
        """
        Initialize the dataset.

        :param meta_path: Path to the metadata file
        :param save_folder: Directory to save generated data
        """
        super().__init__()
        self.meta_path = meta_path
        self.save_folder = save_folder
        try:
            self.data = load_from_data_path(meta_path)
            logging.info(f"Loaded {len(self.data)} items from {self.meta_path}")
        except Exception as e:
            logging.error(f"Failed to load data from {self.meta_path}: {e}")
            raise

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get an item by index.

        :param index: Index of the item
        :return: Data item as a dictionary
        """
        return self.data[index]

    def __len__(self) -> int:
        """
        Get the total number of items.

        :return: Length of the dataset
        """
        return len(self.data)


class ImageCaptionerDataset(DataFlowDataset):
    """
    Dataset for image captioning tasks.
    """
    def __init__(self, meta_path: str, save_folder: str, **kwargs):
        super().__init__(meta_path, save_folder)


class ImageGeneratorDataset(DataFlowDataset):
    """
    Dataset for image generation tasks.
    """
    def __init__(self, meta_path: str, save_folder: str, **kwargs):
        super().__init__(meta_path, save_folder)
        self.generated_folder = os.path.join(save_folder, 'generated')
        os.makedirs(self.generated_folder, exist_ok=True)
        # Update image paths to absolute
        for item in self.data:
            raw_image = item.get('raw_image')
            if raw_image:
                item['image'] = os.path.abspath(os.path.join(self.generated_folder, raw_image))
            else:
                logging.warning("Item missing 'raw_image' key.")


class VideoCaptionerDataset(DataFlowDataset):
    """
    Dataset for video captioning tasks.
    """
    def __init__(self, meta_path: str, save_folder: str, **kwargs):
        super().__init__(meta_path, save_folder)


class VideoGeneratorDataset(DataFlowDataset):
    """
    Dataset for video generation tasks.
    """
    def __init__(self, meta_path: str, save_folder: str, **kwargs):
        super().__init__(meta_path, save_folder)
        self.generated_folder = os.path.join(save_folder, 'generated')
        os.makedirs(self.generated_folder, exist_ok=True)
        # Update video paths to absolute
        for item in self.data:
            raw_video = item.get('raw_video')
            if raw_video:
                item['video'] = os.path.abspath(os.path.join(self.generated_folder, raw_video))
            else:
                logging.warning("Item missing 'raw_video' key.")


class TextGeneratorDataset(DataFlowDataset):
    """
    Dataset for text generation tasks.
    """
    def __init__(self, meta_path: str, save_folder: str, **kwargs):
        super().__init__(meta_path, save_folder)