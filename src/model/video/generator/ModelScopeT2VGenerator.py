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

        outputs = [self.generate_video(prompt) for prompt in prompts]
        return outputs    
    
    def generate_video(self,
                     prompt
                     ):
        output = self.pipeline(prompt['prompt'])
        frames = output.frames[0]
        export_to_video(frames, "modelscopet2v.mp4", fps=10)
        return {"id": prompt["id"], "output": output}