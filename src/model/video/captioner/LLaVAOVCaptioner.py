import numpy as np
import logging
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from tqdm import tqdm
import warnings
from decord import VideoReader, cpu
from src.utils.registry import CAPTIONER_REGISTRY
warnings.filterwarnings("ignore")
device = "cuda"
device_map = "auto"

@CAPTIONER_REGISTRY.register()
class LLaVAOVCaptioner:
    def __init__(self, 
                model_path: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf", 
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
        logging.info(f"VLLM model LLaVAOVCaptioner will initialize with model_path: {model_path}")
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_path, 
                                                                            torch_dtype=torch.float16, 
                                                                            device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.prompt = prompt
        self.max_model_len = max_model_len


    # Function to extract frames from video
    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx)
        spare_frames = spare_frames.asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def generate_batch(self, videos):
        outputs = []
        for video in tqdm(videos, desc="Processing videos"):
            video = self.load_video(video, 16)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": self.prompt},
                    ],
                },
            ]

            prompt = self.processor.apply_chat_template(conversation, 
                                                        add_generation_prompt=True)
            inputs = self.processor(videos=list(video), 
                                    text=prompt, 
                                    return_tensors="pt").to("cuda:0", torch.float16)

            out = self.model.generate(**inputs, 
                                      max_new_tokens=self.max_model_len, pad_token_id=151645, eos_token_id=151645)
            response = self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            response = response[0].split("assistant\n")[-1]
            outputs.append(response)
        return outputs