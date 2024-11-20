import logging
from openai import OpenAI
import base64
from PIL import Image

from src.utils.registry import CAPTIONER_REGISTRY

@CAPTIONER_REGISTRY.register()
class GPTCaptioner:
    def __init__(self, 
                model: str = "gpt-4o", 
                api_key: str = None,
                api_base: str = "https://api.openai.com",
                max_num_seqs: int = 256, 
                temperature: float = 0.2,
                sys_prompt: str = "You are a helpful assistant.",
                prompt: str = "What is the content of this image?",
                **kwargs,
                ):
        logging.info(f"VLLM model GPTCaptioner will initialize.")
        self.model = model
        self.sys_prompt = sys_prompt
        self.prompt = prompt
        self.max_num_seqs = max_num_seqs
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key)

    def encode_images(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_batch(self, images):
        outputs = []
        for image_path in images:
            base64_image = self.encode_images(image_path)
            image_type = image_path.split('.')[-1]
            chat_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": f'{self.sys_prompt}'
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f'{self.prompt}'
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_type};base64,{base64_image}"
                                }
                            },
                        ]
                    },
                ],
                max_tokens=self.max_num_seqs,
                temperature=self.temperature,
            )
            outputs.append(chat_response.choices[0].message.content)

        return outputs

    