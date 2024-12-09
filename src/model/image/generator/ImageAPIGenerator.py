import torch, logging, requests, json, time
from PIL import Image
from openai import OpenAI
from io import BytesIO
from huggingface_hub import InferenceClient

from src.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class ImageAPIGenerator:
    def __init__(
        self,
        model: str = 'stable-diffusion',
        api_key: str = None,
        api_url: str = 'https://modelslab.com/api/v6/realtime/text2img',
        height: int = 1024,
        width: int = 1024,
        quality: str = 'standard', # for openai
        retry_count: int = 10, # for flux
        retry_gap: float = 0.3, # for flux
    ):
        logging.info('ImageAPIGenerator will be initialized')
        self.model = model
        self.api_key = api_key
        self.api_url = api_url
        self.height = height
        self.width = width
        self.quality = quality
        self.retry_count = retry_count
        self.retry_gap = retry_gap
        if 'openai' in self.model:
            self.client = OpenAI(api_key=api_key)
        elif 'huggingface' in self.api_key:
            self.client = InferenceClient(model.split(':')[-1], api_key=api_key)
    
    def generate_batch_openai(self, captions):
        outputs = []
        for caption in captions:
            response = self.client.images.generate(
                model=self.model.split(':')[-1],
                prompt=caption,
                size=f'{self.height}x{self.width}',
                quality=self.quality,
                n=1,
            )
            image_url = response.data[0].url
            outputs.append(image_url)
        return outputs

    def generate_batch_stable_diffusion(self, captions):
        # https://docs.modelslab.com/image-generation/realtime-stable-diffusion/text2img
        outputs = []
        for caption in captions:
            headers = {'Content-Type': 'application/json'}
            payload = json.dumps({
                'key': self.api_key,
                'prompt': caption,
                'output_format': 'img',
                'width': self.width,
                'height': self.height,
                "safety_checker": False,
                "seed": None,
                "samples":1,
                "base64":False,
                "webhook": None,
                "track_id": None
            })
            try:
                response = requests.post(self.api_url, headers=headers, data=payload)
            except Exception as e:
                logging.info(f'request to stable-diffusion failed {e}')
            outputs.append(response.text['proxy_links'])

        return outputs

    def generate_batch_flux(self, captions):
        # https://docs.bfl.ml/quick_start/gen_image/
        outputs = []
        for caption in captions:
            headers = {
                'accept': 'application/json',
                'x-key': self.api_key,
                'Content-Type': 'application/json',
            }
            payload = {
                'prompt': caption,
                'width': self.width,
                'height': self.height,
            }
            try:
                response = requests.post(self.api_url, headers=headers, json=payload).json()
            except Exception as e:
                logging.info(f'request to flux failed {e}')
            result_id, result_url = response['id'], ''
            for _ in range(self.retry_count):
                time.sleep(self.retry_gap)
                try:
                    response = requests.get(
                        'https://api.bfl.ml/v1/get_result',
                        headers={
                            'accept': 'application/json',
                            'x-key': self.api_key,
                        },
                        params={
                            'id': result_id,
                        },
                    ).json()
                    if response['status'] == 'Ready':
                        result_url = response['result']['sample']
                    else:
                        logging.info('image not ready, retry')
                except Exception as e:
                    logging.info(f'request to flux failed {e}')
            if not result_url:
                logging.info('image not ready, exceed retry count, failed')
                continue
            try:
                # signed urls are only valid for 10 minutes
                # so get it immediately
                response = requests.get(result_url)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    outputs.append(image)
                else:
                    logging.info(f'unknown error occurred when getting image')
            except Exception as e:
                logging.info(f'unknown error occurred when getting image, {e}')

        return outputs
    
    def generate_batch_huggingface(self, captions):
        # https://huggingface.co/docs/api-inference/tasks/text-to-image#text-to-image
        outputs = []
        for caption in captions:
            response = self.client.text_to_image(caption)
            outputs.append(response)
        
        return outputs
    
    def generate_batch(self, captions):
        if 'openai' in self.model:
            return self.generate_batch_openai(captions)
        elif 'stable-diffusion' in self.model:
            return self.generate_batch_stable_diffusion(captions)
        elif 'flux' in self.model:
            return self.generate_batch_flux(captions)
        elif 'huggingface' in self.model:
            return self.generate_batch_huggingface(captions)