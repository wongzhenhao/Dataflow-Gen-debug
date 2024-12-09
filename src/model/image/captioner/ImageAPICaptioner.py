import logging, os, base64
from vllm import LLM, SamplingParams
from PIL import Image
from openai import OpenAI
from anthropic import Anthropic
from huggingface_hub import InferenceClient

from src.utils.registry import CAPTIONER_REGISTRY

@CAPTIONER_REGISTRY.register()
class ImageAPICaptioner:
    def __init__(
        self,
        model: str = "openai:gpt-4o",
        api_key: str = None,
        base_url: str = "https://api.openai.com",
        max_tokens: int = 256, 
        temperature: float = 0.2,
        sys_prompt: str = "You are a helpful assistant.",
        prompt: str = "What is the content of this image?",
        **kwargs,
    ):
        logging.info(f"VLLM model ImageAPICaptioner will initialize.")
        self.model = model
        self.sys_prompt = sys_prompt
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        if 'openai' in self.model:
            self.client = OpenAI(api_key=api_key)
        elif 'claude' in self.model:
            self.client = Anthropic(api_key=api_key)
        elif 'huggingface' in self.model:
            self.client = InferenceClient(token=api_key)
        else:
            logging.info(f'model {self.model} not supported')
            raise Exception(f'model {self.model} not supported')
    
    def encode_images(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        # it seems that image type needed by these clients are all the same
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def generate_batch_openai(self, images):
        outputs = []
        for image_path in images:
            base64_image = self.encode_images(image_path)
            image_type = image_path.split('.')[-1]
            message = self.client.chat.completions.create(
                model=self.model.split(':')[-1],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system",  "content": f'{self.sys_prompt}'},
                    {"role": "user", "content": [
                        {"type": "text", "text": f'{self.prompt}'},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/{image_type};base64,{base64_image}"
                        }},
                    ]},
                ],
            )
            outputs.append(message.choices[0].message.content)
        return outputs

    def generate_batch_claude(self, images):
        # https://docs.anthropic.com/en/docs/build-with-claude/vision#about-the-prompt-examples
        outputs = []
        for image_path in images:
            base64_image = self.encode_images(image_path)
            image_type = image_path.split('.')[-1]
            message = self.client.messages.create(
                model=self.model.split(':')[-1],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.sys_prompt,
                messages=[
                    {'role': 'user', 'content': [
                        {'type': 'image', 'source': {
                            'type': 'base64', 'media_type': f'image/{image_type}', 'data': base64_image
                        }},
                        {'type': 'text', 'text': self.prompt}
                    ]},
                ]
            )
            outputs.append(message.content[0].text)
        return outputs

    def generate_batch(self, images):
        if 'openai' in self.model:
            return self.generate_batch_openai(images)
        elif 'claude' in self.model:
            return self.generate_batch_claude(images)
        elif 'huggingface' in self.model:
            # https://huggingface.co/docs/huggingface_hub/guides/inference#openai-compatibility
            # huggingface api has compatibility with openai api
            # so we just need to reuse method
            return self.generate_batch_openai(images)
            