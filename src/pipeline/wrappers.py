# src/pipeline/wrappers.py

import os
import logging
from torch.utils.data import DataLoader
from typing import Any, List, Dict
from PIL import Image
from src.utils.registry import CAPTIONER_REGISTRY, GENERATOR_REGISTRY
from src.data.Dataset import (
    ImageCaptionerDataset, ImageGeneratorDataset,
    VideoCaptionerDataset, VideoGeneratorDataset,
    TextGeneratorDataset
)
from src.pipeline.steps import PipelineStep
from src.data.DataManager import Recorder

from diffusers.utils import export_to_gif, export_to_video

class ImageCaptionerWrapper(PipelineStep):
    """
    Wrapper for image captioning models.
    """
    def __init__(self, name: str, config: dict, data_manager):
        super().__init__(name, config)
        self.data_manager = data_manager
        self.model_name = name.split('_')[-1]

    def execute(self, input_data: str) -> str:
        """
        Execute image captioning on the input data.

        :param input_data: Path to the input data
        :return: Path to the saved captions
        """
        recorder = Recorder(
            data_manager=self.data_manager,
            step_name=self.name,
        )
        dataset = ImageCaptionerDataset(
            meta_path=input_data,
            save_folder=self.data_manager.get_path(self.name),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            collate_fn=self.collate_fn
        )
        logging.info(f"Starting Image Captioning with model {self.model_name}")
        model = CAPTIONER_REGISTRY.get(self.model_name)(**self.config)

        for metadata_batch in dataloader:
            images = [meta['image'] for meta in metadata_batch]
            captions = self.generate_batch(model, images)
            for meta, caption in zip(metadata_batch, captions):
                meta['text'] = caption
            recorder.record(metadata_batch)

        return recorder.dump()

    def generate_batch(self, model, images: List[str]) -> List[str]:
        """
        Generate captions for a batch of images.

        :param model: Captioning model
        :param images: List of image paths
        :return: List of captions
        """
        return model.generate_batch(images)

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for DataLoader.

        :param batch: List of data items
        :return: Batch of metadata
        """
        return batch


class ImageGeneratorWrapper(PipelineStep):
    """
    Wrapper for image generation models.
    """
    def __init__(self, name: str, config: dict, data_manager):
        super().__init__(name, config)
        self.data_manager = data_manager
        self.model_name = name.split('_')[-1]

    def execute(self, input_data: str) -> str:
        """
        Execute image generation based on input prompts.

        :param input_data: Path to the input data
        :return: Path to the saved generated images
        """
        recorder = Recorder(
            data_manager=self.data_manager,
            step_name=self.name
        )
        dataset = ImageGeneratorDataset(
            meta_path=input_data,
            save_folder=self.data_manager.get_path(self.name),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 1),
            shuffle=False,
            collate_fn=self.collate_fn
        )
        logging.info(f"Starting Image Generation with model {self.model_name}")
        model = GENERATOR_REGISTRY.get(self.model_name)(**self.config)

        for metadata_batch in dataloader:
            prompts = [meta['text'] for meta in metadata_batch]
            image_paths = [meta['image'] for meta in metadata_batch]
            generated_images = self.generate_batch(model, prompts)
            for image, path in zip(generated_images, image_paths):
                self.save_image(image, path)

            recorder.record(metadata_batch)

        return recorder.dump()

    def generate_batch(self, model, prompts: List[str]) -> List[Image.Image]:
        """
        Generate images based on a batch of prompts.

        :param model: Image generation model
        :param prompts: List of text prompts
        :return: List of generated PIL Image objects
        """
        return model.generate_batch(prompts)

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for DataLoader.

        :param batch: List of data items
        :return: Batch of metadata
        """
        return batch

    @staticmethod
    def save_image(image: Image.Image, path: str):
        """
        Save the generated image to the specified path.

        :param image: PIL Image object
        :param path: Path to save the image
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)


class VideoCaptionerWrapper(PipelineStep):
    """
    Wrapper for video captioning models.
    """
    def __init__(self, name: str, config: dict, data_manager):
        super().__init__(name, config)
        self.data_manager = data_manager
        self.model_name = name.split('_')[-1]

    def execute(self, input_data: str) -> str:
        """
        Execute video captioning on the input data.

        :param input_data: Path to the input video data
        :return: Path to the saved captions
        """
        recorder = Recorder(
            data_manager=self.data_manager,
            step_name=self.name,
        )
        dataset = VideoCaptionerDataset(
            meta_path=input_data,
            save_folder=self.data_manager.get_path(self.name),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 4),
            shuffle=False,
            collate_fn=self.collate_fn
        )
        logging.info(f"Starting Video Captioning with model {self.model_name}")
        model = CAPTIONER_REGISTRY.get(self.model_name)(**self.config)

        for metadata_batch in dataloader:
            videos = [meta['video'] for meta in metadata_batch]
            try:
                captions = self.generate_batch(model, videos)
                for meta, caption in zip(metadata_batch, captions):
                    meta['text'] = caption
                recorder.record(metadata_batch)
            except Exception as e:
                logging.error(f"Failed to generate video captions: {e}")
                raise

        return recorder.dump()

    def generate_batch(self, model, videos: List[str]) -> List[str]:
        """
        Generate captions for a batch of videos.

        :param model: Video captioning model
        :param videos: List of video paths
        :return: List of captions
        """
        return model.generate_batch(videos)

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for DataLoader.

        :param batch: List of tuples (video, metadata)
        :return: Tuple of list of videos and list of metadata
        """
        return batch


class VideoGeneratorWrapper(PipelineStep):
    """
    Wrapper for video generation models.
    """
    def __init__(self, name: str, config: dict, data_manager):
        super().__init__(name, config)
        self.data_manager = data_manager
        self.model_name = name.split('_')[-1]

    def execute(self, input_data: str) -> str:
        """
        Execute video generation based on input prompts.

        :param input_data: Path to the input prompts data
        :return: Path to the saved generated videos
        """
        recorder = Recorder(
            data_manager=self.data_manager,
            step_name=self.name,
        )
        dataset = VideoGeneratorDataset(
            meta_path=input_data,
            save_folder=self.data_manager.get_path(self.name),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 1),
            shuffle=False,
            collate_fn=self.collate_fn
        )
        logging.info(f"Starting Video Generation with model {self.model_name}")
        model = GENERATOR_REGISTRY.get(self.model_name)(**self.config)

        for metadata_batch in dataloader:
            prompts = [meta['text'] for meta in metadata_batch]
            video_paths = [meta['video'] for meta in metadata_batch]
            try:
                generated_videos = self.generate_batch(model, prompts)
                for video, path in zip(generated_videos, video_paths):
                    self.save_video(video, path)
                recorder.record(metadata_batch)
            except Exception as e:
                logging.error(f"Failed to generate videos: {e}")
                raise

        return recorder.dump()

    def generate_batch(self, model, prompts: List[str]):
        """
        Generate videos based on a batch of prompts.

        :param model: Video generation model
        :param prompts: List of text prompts
        :return: List of generated VideoClip objects
        """
        return model.generate_batch(prompts)

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for DataLoader.

        :param batch: List of data items
        :return: Batch of metadata
        """
        return batch

    @staticmethod
    def save_video(video, path: str):
        """
        Save the generated video to the specified path.

        :param video: MoviePy VideoClip object
        :param path: Path to save the video
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if path.lower().endswith('.gif'):
            export_to_gif(video, path)
        elif path.lower().endswith(('.mp4', '.avi', '.mov')):
            export_to_video(video, path, fps=10)  # Adjust fps as needed
        else:
            logging.warning(f"Unsupported video format for path: {path}")

class TextGeneratorWrapper(PipelineStep):
    """
    Wrapper for text generation models.
    """
    def __init__(self, name: str, config: dict, data_manager):
        """
        Initialize the TextGeneratorWrapper.

        :param name: Name of the pipeline step
        :param config: Configuration dictionary for the step
        :param data_manager: Instance of DataManager for handling data storage
        """
        super().__init__(name, config)
        self.data_manager = data_manager
        self.model_name = name.split('_')[-1]

    def execute(self, input_data):
        """
        Execute text generation based on input prompts.

        :param input_data: Path to the input prompts data (JSON or JSONL)
        :return: Path to the saved generated texts
        """
        recorder = Recorder(
            data_manager=self.data_manager,
            save_folder=self.data_manager.get_path(self.name),
        )
        dataset = TextGeneratorDataset(
            meta_prompt_path=input_data,
            save_folder=self.save_folder
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            collate_fn=self.collate_fn
        )
        logging.info(f"Starting Text Generation with model '{self.model_name}'")
        model = GENERATOR_REGISTRY.get(self.model_name)(**self.config)

        for metadata_batch in dataloader:
            prompts = [meta['text'] for meta in metadata_batch]
            try:
                generated_texts = self.generate_batch(model, prompts)
                for meta, text in zip(metadata_batch, generated_texts):
                    meta['text'] = text
                recorder.record(metadata_batch)
            except Exception as e:
                logging.error(f"Failed to generate texts: {e}")
                raise

        return recorder.dump()

    def generate_batch(self, model, texts: List[str]) -> List[str]:
        """
        Generate texts based on a batch of texts.

        :param model: Text generation model instance
        :param texts: List of text
        :return: List of generated texts
        """
        return model.generate_batch(texts)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        """
        Collate function for DataLoader.

        :param batch: List of data items
        :return: Batch of metadata
        """
        return batch