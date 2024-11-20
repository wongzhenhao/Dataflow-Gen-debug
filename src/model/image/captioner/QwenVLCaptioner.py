import logging
from vllm import LLM, SamplingParams
from PIL import Image

from src.utils.registry import CAPTIONER_REGISTRY

@CAPTIONER_REGISTRY.register()
class QwenVLCaptioner:
    def __init__(self, 
                model_path: str = "Qwen/Qwen2-VL-7B-Instruct", 
                trust_remote_code: bool = True,
                tensor_parallel_size: int = 1,
                max_model_len: int = 256,
                max_num_seqs: int = 128, 
                temperature: float = 0.6,
                top_p: float = 0.9,
                top_k: int = 50,
                repetition_penalty: float = 1.2,
                prompt: str = "What is the content of this image?",
                **kwargs,
                ):
        logging.info(f"VLLM model QwenVLCapioner will initialize with model_path: {model_path}")
        self.model = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len, max_num_seqs=max_num_seqs)
        self.sampling_params = SamplingParams(temperature=temperature,
                                            top_p=top_p,
                                            top_k=top_k,
                                            max_tokens=max_model_len,
                                            )
        self.prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                       "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                       f"{prompt}<|im_end|>\n"
                       "<|im_start|>assistant\n")


    def encode_images(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path).convert('RGB')
        return image


    def generate_batch(self, images):
        inputs, outputs = [], []
        for image in images:
            inputs.append({
                "prompt": self.prompt,
                "multi_modal_data": {
                    "image": self.encode_image(image),
                },
            })
        response = self.model.generate(inputs, self.sampling_params)
        for r in response:
            outputs.append(r.outputs[0].text.strip())

        return outputs