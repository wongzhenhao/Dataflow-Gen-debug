from genmo.mochi_preview.pipelines import (
    DecoderModelFactory,
    DitModelFactory,
    MochiSingleGPUPipeline,
    T5ModelFactory,
    linear_quadratic_schedule,
)
import logging
from src.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class MochiGenerator:
    def __init__(self,
                model_path: str= "genmo/mochi-1-preview",
                cpu_offload: bool = True,
                decode_type: str = "tiled_full",
                **kwargs):
        
        logging.info(f"model MochiGenerator will initialize with model_path: {model_path}")
        self.pipeline = MochiSingleGPUPipeline(
            text_encoder_factory=T5ModelFactory(),
            dit_factory=DitModelFactory(
                model_path=f"{model_path}/dit.safetensors", model_dtype="bf16"
            ),
            decoder_factory=DecoderModelFactory(
                model_path=f"{model_path}/encoder.safetensors",
            ),
            cpu_offload=cpu_offload,
            decode_type=decode_type,
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
        export_to_video(frames, "mochi.mp4", fps=10)
        return {"id": prompt["id"], "output": output}