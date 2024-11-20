import logging
from src.utils.registry import GENERATOR_REGISTRY
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import torch

@GENERATOR_REGISTRY.register()
class I2VGenXLGenerator:
    def __init__(self,
                model_path: str = "ali-vilab/i2vgen-xl",
                variant="fp16",
                **kwargs):
        
        logging.info(f"model I2VGenXLGenerato will initialize with model_path: {model_path}")
        self.pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, variant=variant)
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.enable_vae_slicing()
    
    def generate_batch(self,
                    prompts):

        outputs = []
        for prompt in prompts:
            video = self.pipeline(
                prompt=prompt['text_prompt'],
                image=load_image(prompt['image_prompt']).convert("RGB"),
                generator=torch.manual_seed(8888)
            ).frames[0]
        outputs.append(video)
        return outputs    
