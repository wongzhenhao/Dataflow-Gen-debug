import logging
import aisuite as ai
from src.utils.registry import GENERATOR_REGISTRY
from src.utils.data_utils import load_from_data_path
import json
import os
# APIKEY should be set in the environment variable


# 调用API生成文本
@GENERATOR_REGISTRY.register()
class APIGenerator:
    def __init__(self, 
        model_id : str = 'openai:gpt-4o',
        temperature : float = 0.75,
        top_p : float = 1,
        max_tokens : int = 20,
        n : int = 1,
        stream : bool = False,
        stop = None,
        presence_penalty : float = 0,
        frequency_penalty : float = 0,
        logprobs = None,
        prompt : str = "You are a helpful assistant",
    ):
        logging.info(f"API Generator will generate text using {model_id}")
        self.model_id = model_id # must be <provider:modelname>
        # for model on huggingface, use <huggingface:modelname> 
        # (and don't forget provide your huggingface token in the environment variable or in the config file)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.n = n
        self.stream = stream
        self.stop=stop
        self.presence_penalty=presence_penalty
        self.frequency_penalty=frequency_penalty
        self.prompt = prompt
        self.logprobs = logprobs


    
    def generate_batch(self, texts):
        client = ai.Client()
        models = self.model_id.split(',')
        outputs = []
        

        # 遍历dataset
        for text in texts:
            messages = [
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": text}
                ]
            response = client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                    top_p = self.top_p,
                    max_tokens = self.max_tokens,
                    n = self.n,
                    stream = self.stream,
                    stop = self.stop,
                    logprobs = self.logprobs,
                    presence_penalty = self.presence_penalty,
                    frequency_penalty = self.frequency_penalty,
            )
            content = response.choices[0].message.content
            print(content)
            outputs.append(content)
        return outputs


