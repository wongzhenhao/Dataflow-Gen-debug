import logging
from src.utils.registry import GENERATOR_REGISTRY
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import torch

@GENERATOR_REGISTRY.register()
class ModelScopeT2VGenerator:
    def __init__(self,
                model_path: str = "damo-vilab/text-to-video-ms-1.7b",
                variant="fp16",
                **kwargs):
        
        logging.info(f"model ModelScopeT2VGenerator will initialize with model_path: {model_path}")
        self.pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, variant=variant)
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.enable_vae_slicing()
    
    def generate_batch(self,
                    prompts):
        outputs = []
        for prompt in prompts:
            video = self.pipeline(prompt['text']).frames[0]
            outputs.append(video)
        return outputs    