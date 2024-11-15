import logging
from src.utils.registry import GENERATOR_REGISTRY
from diffusers import AutoencoderKLAllegro, AllegroPipeline
from diffusers.utils import export_to_video
import torch

@GENERATOR_REGISTRY.register()
class AllegroGenerator:
    def __init__(self,
                model_path: str = "rhymes-ai/Allegro",
                subfolder: str = "vae",
                num_frames: int = 16,
                num_inference_steps: int=100,
                guidance_scale: float=7.5,
                negative_prompt: str="",
                max_sequence_length: int=512,
                **kwargs):
        
        vae = AutoencoderKLAllegro.from_pretrained(model_path, subfolder=subfolder, torch_dtype=torch.float32)
        self.pipeline = AllegroPipeline.from_pretrained(
            model_path, vae=vae, torch_dtype=torch.bfloat16
        )
        self.pipeline.to("cuda")
        self.pipeline.vae.enable_tiling()
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.max_sequence_length = max_sequence_length
       
    

    def generate_batch(self,
                    prompts):

        outputs = []
        for prompt in prompts:
            video = self.pipeline(
                prompt=prompt['text_prompt'],
                negative_prompt=self.negative_prompt,
                num_frames=self.num_frames,
                guidance_scale=self.guidance_scale,
                max_sequence_length=self.max_sequence_length,
                num_inference_steps=self.num_inference_steps,
                generator=torch.Generator("cuda:0").manual_seed(42),
            ).frames[0]
            outputs.append(video)
        
        return outputs