import logging
from vllm import LLM, SamplingParams
from PIL import Image

from src.utils.registry import CAPTIONER_REGISTRY

@CAPTIONER_REGISTRY.register()
class LLaVANeXTCaptioner:
    def __init__(self, 
                model_path: str = "llava-hf/llava-v1.6-mistral-7b-hf", 
                trust_remote_code: bool = True,
                tensor_parallel_size: int = 1,
                max_model_len: int = 256,
                max_num_seqs: int = 128, 
                temperature: float = 0.6,
                top_p: float = 0.9,
                top_k: int = 50,
                repetition_penalty: float = 1.2,
                prompt: str = "What is shown in this image?",
                **kwargs,
                ):
        logging.info(f"VLLM model LLaVANeXTCaptioner will initialize with model_path: {model_path}")
        self.model = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len, max_num_seqs=max_num_seqs)
        self.sampling_params = SamplingParams(temperature=temperature,
                                            top_p=top_p,
                                            top_k=top_k,
                                            max_tokens=max_model_len,
                                            repetition_penalty=repetition_penalty
                                            )
        self.prompt = "[INST] <image>\n" + prompt + " [/INST]"


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

    