import os
import json
import torch
import logging
from PIL import Image

from .DataFlowDataset import DataFlowDataset
from src.utils.data_utils import load_from_data_path


class ImageCaptionDataset(DataFlowDataset):
    def __init__(self, 
                meta_image_path: str, 
                image_folder: str,
                save_folder: str,
                model_name: str,
                save_per_batch: bool = True,
                ):
        super().__init__()
        logging.info(f"Load ImageCaptionDataset with meta_image_path: {meta_image_path}, image_folder: {image_folder}")
        self.image_folder = image_folder
        self.save_folder = save_folder

        os.makedirs(save_folder, exist_ok=True)
        self.images_data = load_from_data_path(meta_image_path)
        if save_per_batch:
            start_index = self.update_dataset(model_name)
            self.images_data = self.images_data[start_index:]
            logging.info(f"Find save_per_batch={save_per_batch}. The procedure will start from index {start_index}.")
        

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.images_data[idx]['image'])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        img = Image.open(image_path).convert('RGB')
        return img

    def update_dataset(self, model_name):
        save_path = os.path.join(self.save_folder, model_name + '.jsonl')
        start_index = -1
        try:
            with open(save_path, 'r') as f:
                for index, line in enumerate(f):
                    start_index = index + 1
                    if model_name not in json.loads(line):
                        start_index = index
                        break
        except FileNotFoundError:
            start_index = 0

        return start_index