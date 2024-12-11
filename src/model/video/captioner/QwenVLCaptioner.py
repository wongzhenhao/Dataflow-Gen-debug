import logging
from vllm import LLM, SamplingParams
from src.utils import process_vision_info
from transformers import AutoProcessor

import torch
from src.utils.registry import CAPTIONER_REGISTRY

@CAPTIONER_REGISTRY.register()
class Qwen2VLCaptioner:
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
                prompt: str = "Please describe the video in detail.",
                gpu_memory_utilization=0.7,
                **kwargs,
                ):
        logging.info(f"VLLM model QwenVLCapioner will initialize with model_path: {model_path}")
        self.model = LLM(model=model_path, 
                         trust_remote_code=True, 
                         tensor_parallel_size=tensor_parallel_size, 
                         max_num_seqs=max_num_seqs,
                         dtype=torch.bfloat16,
                         gpu_memory_utilization=gpu_memory_utilization,
                         )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.sampling_params = SamplingParams(temperature=temperature,
                                            top_p=top_p,
                                            top_k=top_k,
                                            max_tokens=max_model_len,
                                            stop_token_ids=[],
                                            )
        self.prompt = prompt

    def generate_batch(self, videos):
        inputs, outputs = [], []
        prompts = []
        mm_data_list = []
        for video in videos:
            messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video,
                        "max_pixels": 224 * 224,  # 检查 max_pixels 是否合适
                        # "fps": 1.0,               # 检查 fps 是否符合要求
                        "nframes":64
                    },
                    {
                        "type": "text",
                        "text": self.prompt,
                    },
                ],
            }
            ]
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)

            mm_data = {}

            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            llm_inputs = {
                "prompt": self.prompt,
                "multi_modal_data": mm_data,
            }
            prompts.append(prompt)
            mm_data_list.append(mm_data)
            llm_inputs = [
                {"prompt": prompt, "multi_modal_data": mm_data}
                for prompt, mm_data in zip(prompts, mm_data_list)
            ]
        response = self.model.generate(llm_inputs, sampling_params=self.sampling_params)
        for r in response:
            outputs.append(r.outputs[0].text.strip())

        return outputs