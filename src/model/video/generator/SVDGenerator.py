import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from src.utils.registry import GENERATOR_REGISTRY
import logging

@GENERATOR_REGISTRY.register()
class SVDGenerator:
    def __init__(self,
                model_path: str="stabilityai/stable-video-diffusion-img2vid-xt",
                variant: str="fp16",
                decode_chunk_size=8,
                **kwargs):

        logging.info(f"model SVDGenerator will initialize with model_path: {model_path}")
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, variant=variant
        )
        self.pipeline.enable_model_cpu_offload()
        self.decode_chunk_size = decode_chunk_size

    def generate_batch(self,
                    prompts):
        outputs = []
        for prompt in prompts:
            image = load_image(image=prompt['image_prompt'])
            image = image.resize((1024,576))
            video = self.pipeline(
                image,
                decode_chunk_size=self.decode_chunk_size,
                generator=torch.manual_seed(42),
            ).frames[0]
            outputs.append(video)
        return outputs  