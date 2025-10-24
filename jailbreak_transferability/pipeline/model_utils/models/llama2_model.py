import torch
import functools
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from transformers import BitsAndBytesConfig

from pipeline.model_utils.model_base import ModelBase

# Code from https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/pipeline/model_utils/llama2_model.py

# Llama 2 chat templates are based on
# - https://github.com/centerforaisafety/HarmBench/blob/main/baselines/model_utils.py

LLAMA2_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

LLAMA2_CHAT_TEMPLATE = "<s>[INST] {instruction} [/INST]"

LLAMA2_CHAT_TEMPLATE_WITH_SYSTEM = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST]"

LLAMA2_REFUSAL_TOKS = [306] # 'I'

def format_instruction_llama2_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True
):
    if system is not None:
        if system == "default":
            system = LLAMA2_DEFAULT_SYSTEM_PROMPT
        formatted_instruction = LLAMA2_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system_prompt=system)
    else:
        formatted_instruction = LLAMA2_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_llama2_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True
):
    if outputs is not None:
        prompts = [
            format_instruction_llama2_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_llama2_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    return result

class Llama2Model(ModelBase):
    def _load_model(self, model_path, dtype=torch.float16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto"
        ).eval()

        model.requires_grad_(False)
        return model
    
    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True
        )

        # From: https://github.com/nrimsky/CAA/blob/main/generate_vectors.py
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer
    
    def _get_model_block_modules(self):
        return self.model.model.layers
    
    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_llama2_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=False)