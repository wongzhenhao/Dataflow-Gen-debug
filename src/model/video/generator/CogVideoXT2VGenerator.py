import logging
from src.utils.registry import GENERATOR_REGISTRY
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video, load_image

@GENERATOR_REGISTRY.register()
class CogVideoXT2VGenerator:
    def __init__(self,
                model_path: str="THUDM/CogVideoX-2b",
                num_videos_per_prompt=1,
                num_inference_steps=50,
                num_frames=49,
                guidance_scale=6,
                **kwargs):
        
        logging.info(f"model CogVideoXT2VGenerator will initialize with model_path: {model_path}")
        self.pipeline = CogVideoXPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.enable_sequential_cpu_offload()
        self.pipeline.vae.enable_slicing()
        self.pipeline.vae.enable_tiling()
        self.num_videos_per_prompt=num_videos_per_prompt
        self.num_inference_steps=num_inference_steps
        self.num_frames=num_frames
        self.guidance_scale=guidance_scale

    def generate_batch(self,
                    prompts):

        outputs = []
        for prompt in prompts:
            video = self.pipeline(
                prompt=prompt['text_prompt'],
                num_videos_per_prompt=self.num_videos_per_prompt,
                num_inference_steps=self.num_inference_steps,
                num_frames=self.num_frames,
                guidance_scale=self.guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]
            outputs.append(video)  
        return outputs  
    