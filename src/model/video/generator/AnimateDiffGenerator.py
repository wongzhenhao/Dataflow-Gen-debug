import logging
from src.utils.registry import GENERATOR_REGISTRY
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, export_to_video
import torch

@GENERATOR_REGISTRY.register()
class AnimateDiffGenerator:
    def __init__(self,
                model_path: str = "emilianJR/epiCRealism",
                cpu_offload: bool = True,
                decode_type: str = "tiled_full",
                **kwargs):
        
        logging.info(f"model AnimateDiffGenerator will initialize with model_path: {model_path}")
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
        self.pipeline = AnimateDiffPipeline.from_pretrained(model_path, motion_adapter=adapter, torch_dtype=torch.float16)
        self.scheduler = DDIMScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        self.pipeline.scheduler = self.scheduler
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_model_cpu_offload()
    

    def generate_batch(self,
                    prompts,
                    num_frames: int=16,
                    num_inference_steps: int=59,
                    guidance_scale: float=7.5):

        outputs = [self.generate_video(prompt,num_frames,num_inference_steps,guidance_scale)
                    for prompt in prompts]
        return outputs
    
    def generate_video(self,
                    prompt,
                    num_frames: int=16,
                    num_inference_steps: int=59,
                    guidance_scale: float=7.5,
                    
                    ):
        output = self.pipeline(
            prompt=prompt['prompt'],
            negative_prompt="",
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(49),
        )
        frames = output.frames[0]
        export_to_gif(frames, "animation.gif")
        # save_video(frames, "animation.mp4")
        return {"id": prompt["id"], "output": output}