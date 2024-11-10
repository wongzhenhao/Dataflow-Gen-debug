import os
import json
import torch
import logging
from src.utils.data_utils import load_from_data_path
from .DataFlowDataset import DataFlowDataset


class VideoDataset(DataFlowDataset):
    def __init__(self, 
                meta_prompt_path: str, 
                save_folder: str,
                model_name: str,
                save_per_batch: bool = True,
                ):
        super().__init__()
        logging.info(f"Load VideoDataset with meta_prompt_path: {meta_prompt_path}")
        self.save_folder = save_folder
        # load prompts data
        self.prompts_data = load_from_data_path(meta_prompt_path)
        os.makedirs(save_folder, exist_ok=True)
        

    def __len__(self):
        return len(self.prompts_data)

    def __getitem__(self, idx):
        return self.prompts_data[idx]

    def update_dataset(self, model_name):
        pass