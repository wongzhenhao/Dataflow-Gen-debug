import logging
from src.utils.registry import GENERATOR_REGISTRY
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

@GENERATOR_REGISTRY.register()
class CogVideoXGenerator:
    def __init__(self,
                model_path: str="THUDM/CogVideoX-5b-I2V",
                num_videos_per_prompt=1,
                num_inference_steps=50,
                num_frames=49,
                guidance_scale=6,
                **kwargs):
        
        logging.info(f"model CogVideoXGenerator will initialize with model_path: {model_path}")
        self.pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
        self.pipeline.vae.enable_tiling()
        self.pipeline.vae.enable_slicing()
        self.num_videos_per_prompt = 1
        self.num_inference_steps = 50
        self.num_frames = 49
        self.guidance_scale = 6

    def generate_batch(self,
                    prompts):
        outputs = []
        for prompt in prompts:
            video = self.pipeline(
                prompt = prompt['text'],
                image = load_image(image=prompt['image_path']),
                num_videos_per_prompt=self.num_videos_per_prompt,
                num_inference_steps=self.num_inference_steps,
                num_frames=self.num_frames,
                guidance_scale=self.guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(42)
            ).frames[0]
            outputs.append(video)
        return outputs 
