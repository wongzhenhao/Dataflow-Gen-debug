import os
import json
import logging

from src.data.VideoDataset import VideoDataset
from src.utils.registry import GENERATOR_REGISTRY
from torch.utils.data import DataLoader
# from genmo.lib.utils import save_video
from diffusers.utils import export_to_gif, export_to_video

class GeneratorWrapper:
    def __init__(self, 
                meta_prompt_path: str,
                save_folder: str,
                ):
        self.meta_prompt_path = meta_prompt_path
        self.save_folder = save_folder
        
    def generate_for_one_model(self, model_name, model_config, batch_size=2, repeat_time=1):
        dataset = VideoDataset(meta_prompt_path=self.meta_prompt_path, save_folder=self.save_folder, model_name=model_name)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        logging.info(f"Start generating videos with model {model_name}")
        model = GENERATOR_REGISTRY.get(model_name)(**model_config)
        save_folder = os.path.join(self.save_folder, model_name)
        os.makedirs(save_folder, exist_ok=True)
        for prompts in dataloader:
            outputs = self.generate_batch(model, prompts, repeat_time)
            for output in outputs:
                if model_name=="AnimateDiffGenerator":
                    save_path = os.path.join(save_folder, f"{output['id']}.gif")
                    export_to_gif(output['output'].frames[0], save_path)
                else:
                    save_path = os.path.join(save_folder, f"{output['id']}.mp4")
                    export_to_video(output['output'].frames[0], save_path, fps=10)
            
                    

    def generate_batch(self, model, prompts, repeat_time):
        repeated_prompts = [prompt for prompt in prompts for _ in range(repeat_time)]
        videos = model.generate_batch(repeated_prompts)
        return videos
    
    def collate_fn(self, batch):
        videos = [item for item in batch]
        return videos
        

        
