import os
import json
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

from src.utils.data_utils import load_from_data_path

class DataFlowDataset(Dataset):
    def __init__(self, args=None):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

class ImageCaptionerDataset(DataFlowDataset):
    def __init__(self, 
                meta_image_path: str, 
                image_folder: str,
                save_folder: str,
                **kwargs,
                ):
        super().__init__()
        logging.info(f"Load ImageCaptionDataset with meta_image_path: {meta_image_path}, image_folder: {image_folder}")
        self.image_folder = image_folder
        self.save_folder = save_folder

        os.makedirs(save_folder, exist_ok=True)
        self.images_data = load_from_data_path(meta_image_path)
        
    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.images_data[idx]['image'])
        return image_path, self.images_data[idx]['id']

class ImageGeneratorDataset(DataFlowDataset):
    def __init__(self, 
                meta_prompt_path: str, 
                save_folder: str,
                **kwargs,
                ):
        super().__init__()
        logging.info(f"Load ImageGeneratorDataset with meta_prompt_path: {meta_prompt_path}")
        self.save_folder = save_folder
        # load prompts data
        self.prompts_data = load_from_data_path(meta_prompt_path)
        os.makedirs(save_folder, exist_ok=True)

    def __len__(self):
        return len(self.prompts_data)

    def __getitem__(self, idx):
        return self.prompts_data[idx]['text'], self.prompts_data[idx]['image']

class VideoCaptionerDataset(DataFlowDataset):
    def __init__(self, 
                meta_video_path: str, 
                video_folder: str,
                save_folder: str,
                **kwargs,
                ):
        super().__init__()
        logging.info(f"Load VideoCaptionDataset with meta_video_path: {meta_video_path}, video_folder: {video_folder}")
        self.video_folder = video_folder
        self.save_folder = save_folder

        os.makedirs(save_folder, exist_ok=True)
        self.videos_data = load_from_data_path(meta_video_path)
        
    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.images_data[idx]['image'])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path).convert('RGB')
        return image, images_data[idx]['id']

class VideoGeneratorDataset(DataFlowDataset):
    def __init__(self, 
                meta_prompt_path: str, 
                save_folder: str,
                **kwargs,
                ):
        super().__init__()
        logging.info(f"Load VideoGeneratorDataset with meta_prompt_path: {meta_prompt_path}")
        self.save_folder = save_folder
        # load prompts data
        self.prompts_data = load_from_data_path(meta_prompt_path)
        os.makedirs(save_folder, exist_ok=True)

    def __len__(self):
        return len(self.prompts_data)

    def __getitem__(self, idx):
        return self.prompts_data[idx]