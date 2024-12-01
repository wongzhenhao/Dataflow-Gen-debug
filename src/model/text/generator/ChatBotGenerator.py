from utils.configs_loader import load_config
from utils.llm_swarm_utils import LLMSwarm, LLMSwarmConfig
from transformers import AutoTokenizer
from huggingface_hub import AsyncInferenceClient
import asyncio
import os
import json
from datasets import load_dataset
from src.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class ChatbotTextGenerator:
    def __init__(self, config):
        """
        使用配置文件初始化 ChatbotTextGenerator。

        Args:
            config (dict): 配置参数。
        """
        self.model_id = config["model_id"]
        self.dataset_name = config["dataset_name"]
        self.split = config["split"]
        self.step_size = config["step_size"]
        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # 配置 LLMSwarm
        self.llm_swarm_config = LLMSwarmConfig(
            instances=config["instances"],
            inference_engine="vllm",
            gpus=config["gpus"],
            model=self.model_id,
            slurm_template_path=config["slurm_template_path"],
            per_instance_max_parallel_requests=config["per_instance_max_parallel_requests"],
        )

    def preprocess_chat(self, conversation):
        """
        预处理聊天记录，提取到最后一个助手响应之前的内容。

        Args:
            conversation (list): 对话的完整记录。
        
        Returns:
            list: 处理后的对话记录。
        """
        assistant_turns = [
            turn for turn in conversation if turn["role"] == "assistant"
        ]
        if assistant_turns:
            last_turn = assistant_turns[-1]
            return conversation[: conversation.index(last_turn)]
        return conversation

    async def process_text(self, prompt, client, semaphore):
        """
        使用模型生成响应。

        Args:
            prompt (str): 输入的文本提示。
            client (AsyncInferenceClient): 推理客户端。
            semaphore (asyncio.Semaphore): 并发限制。
        
        Returns:
            str: 模型生成的响应。
        """
        async with semaphore:
            formatted_prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            response = await client.post(
                json={"prompt": formatted_prompt, "max_tokens": 200}
            )
            return response.json()["text"]

    async def process_dataset(self, dataset, start_index, client, semaphore):
        """
        按批次处理数据集，并生成模型响应。

        Args:
            dataset (Dataset): 要处理的数据集。
            start_index (int): 起始索引。
            client (AsyncInferenceClient): 推理客户端。
            semaphore (asyncio.Semaphore): 并发限制。
        
        Returns:
            list: 模型生成的所有响应。
        """
        results = []
        for idx in range(start_index, len(dataset), self.step_size):
            batch = dataset[idx: idx + self.step_size]
            tasks = [
                self.process_text(self.preprocess_chat(example), client, semaphore)
                for example in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # 保存批次结果
            output_file = os.path.join(self.output_dir, f"batch_{idx}.json")
            with open(output_file, "w") as f:
                json.dump(batch_results, f)
        return results

    def generate_dataset(self, dataset_name, split="train"):
        """
        生成数据集。

        Args:
            dataset_name (str): 数据集的名称。
            split (str): 数据集的分割。
        """
        dataset = load_dataset(dataset_name)[split]
        start_index = 0  # 默认从头开始处理

        with LLMSwarm(self.llm_swarm_config) as swarm:
            client = AsyncInferenceClient(model=swarm.endpoint)
            semaphore = asyncio.Semaphore(swarm.suggested_max_parallel_requests)

            async def main():
                print(f"开始处理数据集: {dataset_name}")
                await self.process_dataset(dataset, start_index, client, semaphore)
                print(f"处理完成，结果保存在 {self.output_dir}")

            asyncio.run(main())



if __name__ == "__main__":
    configs = load_config("/mnt/petrelfs/zhaozhengyang/herunming/Dataflow-Gen/configs/textbasic.yaml")
    generator = ChatbotTextGenerator(
        configs
    )
    generator.generate_dataset(dataset_name="lmsys/chatbot_arena_conversations")
