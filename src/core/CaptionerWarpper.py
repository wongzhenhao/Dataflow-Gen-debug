import os
import json
import logging

from src.data.ImageCaptionDataset import ImageCaptionDataset
from src.utils.registry import CAPTION_MODEL_REGISTRY
from torch.utils.data import DataLoader

class CaptionerRecorder():
    def __init__(self, 
                captioner: str,
                save_folder: str,
                save_per_batch: bool = True,
                ):
        self.captioner = captioner
        self.save_folder = save_folder
        self.save_per_batch = save_per_batch
        self.captions = []

    def record_captions(self, captions, repeat_time):
        batch_captions = []
        for index in range(0, len(captions), repeat_time):
            item_caption = {f'{self.captioner}': captions[index:index+repeat_time]}
            batch_captions.append(item_caption)
        
        if self.save_per_batch:
            save_folder = os.path.join(self.save_folder, self.captioner) + '.jsonl'
            with open(save_folder, 'a') as f:
                for item in batch_captions:
                    f.write(json.dumps(item) + '\n')

        self.captions.extend(batch_captions)

    def dump_record(self):
        if not self.save_per_batch:
            save_folder = os.path.join(self.save_folder, self.captioner) + '.jsonl'
            with open(save_folder, 'w') as f:
                for item in self.captions:
                    f.write(json.dumps(item) + '\n')
        logging.info(f"Captions for model {self.captioner} have been saved to folder: {self.save_folder}, total {len(self.captions)} items.")

class CaptionerWarpper:
    def __init__(self, 
                meta_image_path: str,
                image_folder: str,
                save_folder: str,
                save_per_batch: bool = True,
                ):
        self.meta_image_path = meta_image_path
        self.image_folder = image_folder
        self.save_folder = save_folder
        self.save_per_batch = save_per_batch
        
    def generate_for_one_model(self, model_name, repeat_time, batch_size, model_config):
        recorder = CaptionerRecorder(save_folder=self.save_folder, save_per_batch=self.save_per_batch, captioner=model_name)
        dataset = ImageCaptionDataset(meta_image_path=self.meta_image_path, image_folder=self.image_folder, save_per_batch=self.save_per_batch, save_folder=self.save_folder, model_name=model_name)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        logging.info(f"Start generating captions with model {model_name}")
        model = CAPTION_MODEL_REGISTRY.get(model_name)(**model_config)
        for images in dataloader:
            captions = self.generate_batch(model, images, repeat_time)
            recorder.record_captions(captions, repeat_time)
        recorder.dump_record()

    def generate_batch(self, model, images, repeat_time):
        repeated_images = [image for image in images for _ in range(repeat_time)]
        captions = model.generate_batch(repeated_images)
        return captions

    def collate_fn(self, batch):
        images = [img for img in batch]
        return images
        

        
