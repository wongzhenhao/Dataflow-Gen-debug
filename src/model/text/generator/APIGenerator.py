import logging
import aisuite as ai
from src.utils.registry import GENERATOR_REGISTRY
from src.utils.data_utils import load_from_data_path
import json
# APIKEY should be set in the environment variable or provided in the config file


# 调用API生成文本
@GENERATOR_REGISTRY.register()
class APIGenerator:
    def __init__(self, args_dict: dict):
        super().__init__()
        self.Generator_name = "APIGenerator"
        self.model_id = args_dict.get('model_id', 'openai:gpt-4o') # must be <provider:modelname>
        # for model on huggingface, use <huggingface:modelname> 
        # (and don't forget provide your huggingface token in the environment variable or in the config file)
        self.dataset_name = args_dict.get('dataset_name', 'lmsys/chatbot_arena_conversations')
        self.output_dir = args_dict.get('output_dir', './results/text')
        self.output_file_name = args_dict.get('output_file_name', 'api_generated_text.jsonl')
        self.temperature = args_dict.get('temperature', 0.75)
        self.api_key = args_dict.get('api_key', None)

    
    def generate_batch(self, models, texts):
        client = ai.Client()
        models = self.model_id.split(',')
        outputs = []
        

        # 遍历dataset
        for text in texts:
            try:
                data = json.loads(text)
                system_prompt = data.get('system_prompt', '')
                user_prompt = data.get('user_prompt', '')
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            else:
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": text}
                ]
            for model in models:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content
                json_data = {
                    'model': model,
                    'content': content
                }
                outputs.append(json_data)
        return outputs


