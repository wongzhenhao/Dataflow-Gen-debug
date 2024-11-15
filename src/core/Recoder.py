import os
import json
import logging
from diffusers.utils import export_to_gif, export_to_video


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

    def record(self, captions, caption_ids):
        batch_captions = []
        for caption_id, caption in zip(caption_ids, captions):
            item_caption = {"id": caption_id, "caption": caption}
            batch_captions.append(item_caption)
        
        if self.save_per_batch:
            save_folder = os.path.join(self.save_folder, self.captioner) + '.jsonl'
            with open(save_folder, 'a') as f:
                for item in batch_captions:
                    f.write(json.dumps(item) + '\n')

        self.captions.extend(batch_captions)

    def dump(self):
        print(self.captions)
        if not self.save_per_batch:
            save_folder = os.path.join(self.save_folder, self.captioner) + '.jsonl'
            with open(save_folder, 'w') as f:
                for item in self.captions:
                    f.write(json.dumps(item) + '\n')
        logging.info(f"Captions for model {self.captioner} have been saved to folder: {self.save_folder}, total {len(self.captions)} items.")

class GeneratorRecorder():
    def __init__(self, 
                generator: str,
                save_folder: str,
                **kwargs,
                ):
        self.generator = generator
        self.save_folder = save_folder

    def save(self, content, content_path):
        if "jpg" in content_path or "png" in content_path:
            content.save(content_path)
        elif "gif" in content_path:
            export_to_gif(content, content_path)
        elif "mp4" in content_path:
            export_to_video(content, content_path, fps=10)

    def record(self, contents, contents_path):
        save_dir = os.path.join(self.save_folder, self.generator)
        os.makedirs(save_dir, exist_ok=True)
        for content, content_path in zip(contents, contents_path):
            save_path = os.path.join(save_dir, content_path)
            base_dir = os.path.dirname(save_path)
            os.makedirs(base_dir, exist_ok=True)
            self.save(content, save_path)

    def dump(self):
        logging.info(f"Images/Videos for model {self.generator} have been saved to folder: {self.save_folder}")