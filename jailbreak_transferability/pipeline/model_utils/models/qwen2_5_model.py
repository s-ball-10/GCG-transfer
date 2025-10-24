
import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

from pipeline.model_utils.model_base import ModelBase

# Chat templates are based on
# - https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/tokenizer_config.json

QWEN2_5_DEFAULT_SYSTEM_PROMPT = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant."""

QWEN2_5_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN2_5_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

def format_instruction_qwen2_5(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True
):
    if system is not None:
        if system == "default":
            system = QWEN2_5_DEFAULT_SYSTEM_PROMPT
        formatted_instruction = QWEN2_5_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system_prompt=system)
    else:
        formatted_instruction = QWEN2_5_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_qwen2_5(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen2_5(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen2_5(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

class Qwen2_5Model(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        model.requires_grad_(False) 

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_qwen2_5, tokenizer=self.tokenizer, system="default", include_trailing_whitespace=True)

    def _get_model_block_modules(self):
        return self.model.model.layers
