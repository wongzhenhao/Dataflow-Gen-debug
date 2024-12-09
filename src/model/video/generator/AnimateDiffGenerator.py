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
                num_frames: int=16,
                num_inference_steps: int=59,
                guidance_scale: float=7.5,
                negative_prompt: str="",
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
        if cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt

    def generate_batch(self,
                    prompts):

        outputs=[]
        for prompt in prompts:
            print(prompt)
            video = self.pipeline(
                prompt=prompt['text'],
                negative_prompt=self.negative_prompt,
                num_frames=self.num_frames,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(49)
            ).frames[0]
            outputs.append(video)
        return outputs