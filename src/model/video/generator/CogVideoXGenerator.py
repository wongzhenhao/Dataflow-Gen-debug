import logging
from src.utils.registry import GENERATOR_REGISTRY
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

@GENERATOR_REGISTRY.register()
class CogVideoXGenerator:
    def __init__(self,
                model_path: str="THUDM/CogVideoX-5b-I2V",
                **kwargs):
        
        logging.info(f"model CogVideoXGenerator will initialize with model_path: {model_path}")
        self.pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
        self.pipeline.vae.enable_tiling()
        self.pipeline.vae.enable_slicing()

    def generate_batch(self,
                    prompts):
        return outputs  
    
    def generate_video(self,
                    prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=50,
                    num_frames=49,
                    guidance_scale=6,
                    seed: int=42
                    ):
        image = load_image(image=prompt['image_path'])

        output = self.pipeline(
            prompt=prompt['prompt'],
            image=image,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )     
        frames = output.frames[0]
        export_to_video(frames, 'cogvideo.mp4', fps=8)
        return {"id": prompt["id"], "output": output}
    
