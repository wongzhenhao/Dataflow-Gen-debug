from src.utils.registry import GENERATOR_REGISTRY
from src.utils.models.videoldm import VideoLDM
import logging

@GENERATOR_REGISTRY.register()
class VideoLDMGenerator:
    def __init__(self,
                model_path: str = 'CompVis/stable-diffusion-v1-4',
                subfolder: str = 'unet',
                low_cpu_mem_usage: bool = False,
                **kwargs):
        
        logging.info(f"model VideoLDMGenerator will initialize with model_path: {model_path}")
        self.model = VideoLDM.from_pretrained(
            model_path,
            subfolder = subfolder,
            low_cpu_mem_usage = low_cpu_mem_usage
        )
    
    def generate_batch(self,
                    prompts):

        outputs = [self.generate_video(prompt) for prompt in prompts]
        return outputs  
    
    def generate_video(self,
                     prompt,
                     height: int=480,
                     width: int=848,
                     num_frames: int=31,
                     num_inference_steps: int=64,
                     batch_cfg: bool=False,
                     seed: int=12345,
                     ):

        output = self.pipeline(
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            sigma_schedule=linear_quadratic_schedule(64, 0.025),
            cfg_schedule=[4.5] * 64,
            batch_cfg=batch_cfg,
            prompt=prompt['prompt'],
            negative_prompt="",
            seed=seed,
        )
        frames = output.frames[0]
        export_to_video(frames, "videoldm.mp4", fps=10)
        return {"id": prompt["id"], "output": output}