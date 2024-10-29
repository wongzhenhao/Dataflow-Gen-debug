import logging
from vllm import LLM, SamplingParams
from PIL import Image

from src.utils.registry import CAPTION_MODEL_REGISTRY

@CAPTION_MODEL_REGISTRY.register()
class MLLamaCaptioner:
    def __init__(self, 
                model_path: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", 
                trust_remote_code: bool = True,
                tensor_parallel_size: int = 8,
                max_model_len: int = 256,
                max_num_seqs: int = 128, 
                temperature: float = 0.6,
                top_p: float = 0.9,
                top_k: int = 50,
                repetition_penalty: float = None,
                prompt: str = "What is the content of this image?",
                **kwargs,
                ):
        logging.info(f"VLLM model MLLamaCaptioner will initialize with model_path: {model_path}")
        self.model = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len, max_num_seqs=max_num_seqs, enforce_eager=True)
        self.sampling_params = SamplingParams(temperature=temperature,
                                            top_p=top_p,
                                            top_k=top_k,
                                            max_tokens=max_model_len,
                                            )
        self.prompt = "<|image|><|begin_of_text|>" + prompt


    def generate_batch(self, images):
        inputs, outputs = [], []
        for image in images:
            inputs.append({
                "prompt": self.prompt,
                "multi_modal_data": {
                    "image": image
                },
            })
        response = self.model.generate(inputs, self.sampling_params)
        for r in response:
            outputs.append(r.outputs[0].text.strip())

        return outputs

    