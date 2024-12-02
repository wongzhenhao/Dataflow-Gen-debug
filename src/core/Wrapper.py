import os
import logging
from torch.utils.data import DataLoader

from src.utils.registry import CAPTIONER_REGISTRY, GENERATOR_REGISTRY
from .Recoder import CaptionerRecorder, GeneratorRecorder
from src.data.Dataset import ImageCaptionerDataset, ImageGeneratorDataset, TextGeneratorDataset, VideoCaptionerDataset, VideoGeneratorDataset

class ImageCaptionerWrapper:
    def __init__(self, 
                meta_path: str,
                image_folder: str,
                save_folder: str,
                save_per_batch: bool = True,
                ):
        self.meta_image_path = meta_path
        self.image_folder = image_folder
        self.save_folder = save_folder
        self.save_per_batch = save_per_batch
        
    def generate_for_one_model(self, model_name, batch_size, model_config):
        recorder = CaptionerRecorder(save_folder=self.save_folder, save_per_batch=self.save_per_batch, captioner=model_name)
        dataset = ImageCaptionerDataset(meta_image_path=self.meta_image_path, image_folder=self.image_folder, save_folder=self.save_folder)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        logging.info(f"Start generating captions with model {model_name}")
        model = CAPTIONER_REGISTRY.get(model_name)(**model_config)
        for images in dataloader:
            captions = self.generate_batch(model, images[0])
            recorder.record(captions, images[1])
        recorder.dump()

    def generate_batch(self, model, images):
        captions = model.generate_batch(images)
        return captions

    def collate_fn(self, batch):
        images = [image for image, _ in batch], [image_id for _, image_id in batch]
        return images


class ImageGeneratorWrapper:
    def __init__(self, 
                meta_path: str,
                save_folder: str,
                ):
        self.meta_prompt_path = meta_path
        self.save_folder = save_folder
        
    def generate_for_one_model(self, model_name, model_config, batch_size):
        recorder = GeneratorRecorder(save_folder=self.save_folder, generator=model_name)
        dataset = ImageGeneratorDataset(meta_prompt_path=self.meta_prompt_path, save_folder=self.save_folder)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        logging.info(f"Start generating images with model {model_name}")
        model = GENERATOR_REGISTRY.get(model_name)(**model_config)
        save_folder = os.path.join(self.save_folder, model_name)
        os.makedirs(save_folder, exist_ok=True)
        for prompts in dataloader:
            images = self.generate_batch(model, prompts[0])
            recorder.record(images, prompts[1])
        recorder.dump()
                       
    def generate_batch(self, model, prompts):
        images = model.generate_batch(prompts)
        return images
    
    def collate_fn(self, batch):
        images = [prompt for prompt, _ in batch], [path for _, path in batch]
        return images

class TextGeneratorWrapper:
    def __init__(self, 
                meta_path: str,
                save_folder: str,
                save_file: str
                ):
        self.meta_prompt_path = meta_path
        self.save_folder = save_folder
        self.save_file = save_file
        
    def generate_for_one_model(self, model_name, model_config, batch_size):
        recorder = GeneratorRecorder(save_folder=self.save_folder, generator=model_name)
        dataset = TextGeneratorDataset(meta_prompt_path=self.meta_prompt_path, save_folder=self.save_folder)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        logging.info(f"Start generating text with model {model_name}")
        model = GENERATOR_REGISTRY.get(model_name)(**model_config)
        save_folder = os.path.join(self.save_folder, model_name)
        os.makedirs(save_folder, exist_ok=True)
        generated_texts = []
        for prompts in dataloader:
            generated_text = self.generate_batch(model, prompts)
            generated_texts.append(generated_text)
        recorder.record_text(generated_texts, self.save_file)
        recorder.dump()

    def generate_batch(self, model, prompts):
        generated_texts = model.generate_batch(prompts)
        return generated_texts


class VideoCaptionerWrapper:
    def __init__(self, 
                meta_path: str,
                video_folder: str,
                save_folder: str,
                save_per_batch: bool = True,
                ):
        self.meta_video_path = meta_path
        self.video_folder = video_folder
        self.save_folder = save_folder
        self.save_per_batch = save_per_batch
        
    def generate_for_one_model(self, model_name, batch_size, model_config):
        recorder = CaptionerRecorder(save_folder=self.save_folder, save_per_batch=self.save_per_batch, captioner=model_name)
        dataset = VideoCaptionerDataset(meta_video_path=self.meta_video_path, video_folder=self.video_folder, save_folder=self.save_folder)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        logging.info(f"Start generating captions with model {model_name}")
        model = CAPTIONER_REGISTRY.get(model_name)(**model_config)
        for videos in dataloader:
            captions = self.generate_batch(model, videos[0])
            recorder.record(captions, videos[1])
        recorder.dump()

    def generate_batch(self, model, videos):
        captions = model.generate_batch(videos)
        return captions

    def collate_fn(self, batch):
        videos = [video for video, _ in batch], [video_id for _, video_id in batch]
        return videos

class VideoGeneratorWrapper:
    def __init__(self, 
                meta_path: str,
                save_folder: str,
                ):
        self.meta_prompt_path = meta_path
        self.save_folder = save_folder
        
    def generate_for_one_model(self, model_name, model_config, batch_size):
        recorder = GeneratorRecorder(save_folder=self.save_folder, generator=model_name)
        dataset = VideoGeneratorDataset(meta_prompt_path=self.meta_prompt_path, save_folder=self.save_folder, model_name=model_name)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        logging.info(f"Start generating videos with model {model_name}")
        model = GENERATOR_REGISTRY.get(model_name)(**model_config)
        save_folder = os.path.join(self.save_folder, model_name)
        os.makedirs(save_folder, exist_ok=True)
        for prompts in dataloader:
            videos = self.generate_batch(model, prompts[0])
            recorder.record(videos, prompts[1])
        recorder.dump()
            
    def generate_batch(self, model, prompts):
        videos = model.generate_batch(prompts)
        return videos
    
    def collate_fn(self, batch):
        videos = [item['video'] for item in batch]
        prompts = [{'text_prompt':item['text']} for item in batch]
        if 'image_path'in batch[0]:
            prompts = [{'text_prompt':item['text'],'image_prompt':item['image_path']} for item in batch]
        return prompts, videos