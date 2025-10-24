import torch
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
import torch.nn.functional as F
import transformers.generation.utils as gen_utils

from pipeline.utils.hook_utils import add_hooks

# Code from: https://github.com/andyrdt/refusal_direction/blob/main/pipeline/model_utils/model_base.py

class ModelBase(ABC):
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.model: AutoModelForCausalLM = self._load_model(model_name_or_path)
        self.tokenizer: AutoTokenizer = self._load_tokenizer(model_name_or_path)
        self.model_block_modules = self._get_model_block_modules()
        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
    
    def del_model(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model

    def tokenize_texts(self, texts):
        return self.tokenizer(texts, return_tensors='pt', padding=True, truncation=False)

    @abstractmethod
    def _load_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
        pass

    @abstractmethod
    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        pass        

    @abstractmethod
    def _get_tokenize_instructions_fn(self):
        pass

    @abstractmethod
    def _get_model_block_modules(self):
        pass

    def generate_completions(self, texts, fwd_pre_hooks=[], fwd_hooks=[], batch_size=10, max_new_tokens=100):
        if hasattr(gen_utils.DynamicCache, "get_seq_length"):
            gen_utils.DynamicCache.get_max_length = gen_utils.DynamicCache.get_seq_length

        self.tokenizer.padding_side = "left"
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens, 
            do_sample=False)
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        completions = []
        for i in tqdm(range(0, len(texts), batch_size)):
            tokenized_texts = self.tokenize_instructions_fn(instructions=texts[i:i + batch_size])

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                generation_toks = self.model.generate(
                    input_ids=tokenized_texts.input_ids.to(self.model.device),
                    attention_mask = tokenized_texts.attention_mask.to(self.model.device),
                    generation_config=generation_config,
                    use_cache=False 
                )

                for prompt_ids, generation in zip(tokenized_texts.input_ids, generation_toks):
                    prompt_length = prompt_ids.shape[0]  # length of the input prompt tokens
                    new_tokens = generation[prompt_length:]  # slice out only the new generation tokens
                    completions.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

        return completions
    
    def get_top_k_next_token_predictions(self, texts, k=10, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8):
        device = next(self.model.parameters()).device
        top_k_tokens_and_probs = []

        for i in tqdm(range(0, len(texts), batch_size)):
            tokenized_texts = self.tokenize_instructions_fn(instructions=texts[i:i + batch_size])
            tokenized_texts = tokenized_texts.to(device)

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                outputs = self.model(**tokenized_texts)
                logits = outputs.logits
                last_token_logits = logits[:, -1, :]
                probs = F.softmax(last_token_logits, dim=-1)
                topk_probs, topk_ids = torch.topk(probs, k, dim=-1)

                topk_probs = topk_probs.squeeze(0).cpu().tolist()     # [p1, p2, …]
                topk_ids = topk_ids  .squeeze(0).cpu().tolist()     # [id1, id2, …]
                tokens = self.tokenizer.batch_decode(topk_ids)

                # topk_tokens = [self.tokenizer.decode(token_id) for token_id in topk_ids[0]]
                # top_k_tokens_and_probs.extend(list(zip(topk_tokens, topk_probs[0].tolist())))
                top_k_tokens_and_probs.extend(list(zip(tokens, topk_ids, topk_probs)))
                
        return top_k_tokens_and_probs
