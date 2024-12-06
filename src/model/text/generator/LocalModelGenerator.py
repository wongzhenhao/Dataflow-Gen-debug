import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.registry import GENERATOR_REGISTRY
from src.utils.data_utils import load_from_data_path
from src.pipeline.wrappers import TextGeneratorWrapper


@GENERATOR_REGISTRY.register()
class LocalModelGenerator:
    def __init__(self, args_dict: dict):
        super().__init__()
        self.Generator_name = "LocalModelGenerator"
        self.model_path = args_dict.get('model_path', 'qwen/Qwen-7B-Instruct') 
        self.dataset_name = args_dict.get('dataset_name', 'lmsys/chatbot_arena_conversations')
        self.output_dir = args_dict.get('output_dir', './results/text')
        self.output_file_name = args_dict.get('output_file_name', 'local_generated_text.jsonl')
        self.temperature = args_dict.get('temperature', 0.75)
        self.max_length = args_dict.get('max_length', 500)
        self.device = args_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        使用本地模型生成响应。
        """
        # 构建完整的输入
        full_prompt = f"{system_prompt}\n{user_prompt}"
        inputs = self.tokenizer.encode(full_prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=self.max_length,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 去除输入部分，只保留生成的部分
        response = generated_text.split(self.tokenizer.eos_token)[-1].strip()
        return response

    def generate_batch(self, model, texts):
        """
        生成一批响应，并返回结果。
        """
        outputs = []
        

        # 遍历数据集
        for text in texts:
            try:
                data = json.loads(text)
                system_prompt = data.get('system_prompt', '')
                user_prompt = data.get('user_prompt', '')
            except:
                system_prompt = ""
                user_prompt = text
            # 生成响应
            try:
                response = self.generate_response(system_prompt, user_prompt)
                json_data = {
                    'model': model,
                    'system_prompt': system_prompt,
                    'user_prompt': user_prompt,
                    'content': response
                }
                outputs.append(json_data)
            except Exception as e:
                logging.error(f"Error generating response for data {data}: {e}")
        
        return outputs
