import numpy as np
import logging
import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from tqdm import tqdm
import warnings
from decord import VideoReader, cpu
from src.utils.registry import CAPTIONER_REGISTRY
warnings.filterwarnings("ignore")
device = "cuda"
device_map = "auto"

@CAPTIONER_REGISTRY.register()
class LlavaNextVideoCaptioner:
    def __init__(self, 
                model_path: str = "llava-hf/LLaVA-NeXT-Video-7B-hf", 
                trust_remote_code: bool = True,
                tensor_parallel_size: int = 1,
                max_model_len: int = 2048,
                max_num_seqs: int = 128, 
                temperature: float = 0.6,
                top_p: float = 0.9,
                top_k: int = 50,
                repetition_penalty: float = 1.2,
                prompt: str = "Please describe the video in detail.",
                gpu_memory_utilization=0.7,
                **kwargs,
                ):
        logging.info(f"VLLM model LlavaNextVideo will initialize with model_path: {model_path}")
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                        model_path, 
                        torch_dtype=torch.float16, 
                        low_cpu_mem_usage=True, 
                    ).to(device)
        self.processor =processor = LlavaNextVideoProcessor.from_pretrained(model_path)

        self.prompt = prompt
        self.max_model_len = max_model_len

    def load_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), fps)]
        if len(frame_idx) > 32:
            sample_fps = 32
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames

    def generate_batch(self, videos):
        outputs = []
        for video_path in tqdm(videos, desc="Processing videos"):
            video = self.load_video(video_path)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe the video in detail."},
                        {"type": "video"},
                        ],
                },
            ]

            prompt = self.processor.apply_chat_template(conversation, 
                                                        add_generation_prompt=True)
            inputs_video = self.processor(text=prompt, 
                                          videos=video, 
                                          padding=True, 
                                          return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs_video, 
                                         max_new_tokens=self.max_model_len, 
                                         do_sample=False)
            self.processor.decode(output[0][2:], skip_special_tokens=True)

            response = self.processor.decode(output[0][2:], skip_special_tokens=True).split("ASSISTANT: ")[-1]
            outputs.append(response)

        return outputs
