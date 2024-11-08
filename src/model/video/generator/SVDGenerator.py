import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from src.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class SVDGenerator:
    def __init__(self,
                model_path: str="stabilityai/stable-video-diffusion-img2vid-xt",
                **kwargs):

        logging.info(f"model SVDGenerator will initialize with model_path: {model_path}")
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16"
        )
        self.pipeline.enable_model_cpu_offload()

    def generate_batch(self,
                    prompts):

        outputs = [self.generate_video(prompt) for prompt in prompts]
        return outputs  
      
    def generate_video(self,
                     prompt,
                     decode_chunk_size=8,
                     seed=42,
                     ):
        image = load_image(image=prompt["image_path"])
        image = image.resize((1024, 576))

        generator = torch.manual_seed(seed)

        output = self.pipeline(image, decode_chunk_size=decode_chunk_size, generator=generator)     
        video = output.frames[0]
        export_to_video(video, 'svd.mp4', fps=7)
        return {"id": prompt["id"], "output": output}