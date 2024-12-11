import logging
from src.utils.registry import GENERATOR_REGISTRY
from vllm import LLM,SamplingParams
from huggingface_hub import snapshot_download

@GENERATOR_REGISTRY.register()
class LocalModelGenerator:
    def __init__(self,
                 device : str = "cuda",
                 model_path : str = "Qwen/Qwen2.5-0.5B-Instruct",
                 n : int = 1,
                 best_of : int = None,
                 presence_penalty : float = 0,
                 frequency_penalty : float = 0,
                 repetition_penalty : float = 1,
                 temperature : float = 1,
                 top_p : float = 1,
                 top_k : int = -1,
                 min_p : float = 0,
                 seed : int = None,
                 stop = None, # List[str]
                 stop_token_ids = None, #List[str]
                 ignore_eos : bool = False,
                 max_tokens : int = 32,
                 min_tokens : int = 0,
                 logprobs : int = None,
                 prompt_logprobs : int = None,
                 detokenize : bool = True,
                 skip_special_tokens : bool = True,
                 spaces_between_special_tokens : bool = True,
                 logits_processors = None, #Any
                 include_stop_str_in_output : bool = False,
                 truncate_prompt_tokens : int = None,
                 logit_bias = None, # Dict[int,float]
                 allowed_token_ids = None, # List[int]
                 download_dir : str = "ckpr/models/",
                 prompt : str = "You are a helpful assistant",
                 ):
        logging.info(f"Local Model Generator will generate text using model {model_path}")
        self.device = device
        self.prompt = prompt
        self.real_model_path = snapshot_download(repo_id=model_path,local_dir=f"{download_dir}{model_path}")
        self.sampling_params = SamplingParams(
            n = n,
            best_of= best_of,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            temperature= temperature,
            top_p = top_p,
            top_k = top_k,
            min_p = min_p,
            seed = seed,
            stop = stop,
            stop_token_ids= stop_token_ids,
            ignore_eos= ignore_eos,
            max_tokens= max_tokens,
            min_tokens= min_tokens,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            detokenize=detokenize,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            logits_processors=logits_processors,
            include_stop_str_in_output=include_stop_str_in_output,
            truncate_prompt_tokens=truncate_prompt_tokens,
            logit_bias=logit_bias,
            allowed_token_ids=allowed_token_ids,
        )
    
    def generate_batch(self,texts):
        texts = [self.prompt +"\n" + text for text in texts]
        llm = LLM(model=self.real_model_path, device=self.device)
        responses = llm.generate(prompts=texts,sampling_params=self.sampling_params)
        return [output.outputs[0].text for output in responses]
    

    