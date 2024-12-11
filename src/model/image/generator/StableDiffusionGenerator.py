from diffusers import StableDiffusion3Pipeline
import torch, logging
from PIL import Image

from src.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class StableDiffusionGenerator:
    def __init__(self, 
                model_path: str = "stabilityai/stable-diffusion-3.5-large", 
                height: int = 1024,
                weight: int = 1024,
                guidance_scale: float = 4.5,
                num_inference_steps: int = 28,
                max_sequence_length: int = 512,
                device: str = 'cuda',
                **kwargs,
                ):
        logging.info(f"model StableDiffusionGenerator will initialize with model_path: {model_path}")
        self.model = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        self.height, self.width = height, weight
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.max_sequence_length = max_sequence_length

    def generate_batch(self, captions):
        outputs = []
        for caption in captions:
            image = self.model(
                caption,
                height=self.height,
                width=self.width,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                max_sequence_length=self.max_sequence_length
            ).images[0]
            outputs.append(image)

        return outputs

    