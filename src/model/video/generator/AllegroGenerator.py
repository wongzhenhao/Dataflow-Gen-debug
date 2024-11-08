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
                **kwargs):
        
        vae = AutoencoderKLAllegro.from_pretrained(model_path, subfolder=subfolder, torch_dtype=torch.float32)
        self.pipeline = AllegroPipeline.from_pretrained(
            model_path, vae=vae, torch_dtype=torch.bfloat16
        )
        self.pipeline.to("cuda")
        self.pipeline.vae.enable_tiling()
       
    

    def generate_batch(self,
                    prompts,
                    num_frames: int=16,
                    num_inference_steps: int=100,
                    guidance_scale: float=7.5):

        outputs = [self.generate_video(prompt,num_frames,num_inference_steps,guidance_scale)
                    for prompt in prompts]
        return outputs
    
    def generate_video(self,
                    prompt,
                    num_frames: int=16,
                    num_inference_steps: int=100,
                    guidance_scale: float=7.5,
                    device: str="cuda:0"
                    
                    ):
        prompt['prompt'] = prompt['prompt'].format(prompt['prompt'].lower().strip())
        output = self.pipeline(
            prompt=prompt['prompt'],
            negative_prompt="",
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cuda:0").manual_seed(42),
        )
        frames = output.frames[0]
        export_to_video(frames, "demo/allegro.mp4", fps=15)
        # save_video(frames, "animation.mp4")
        return {"id": prompt["id"], "output": output}