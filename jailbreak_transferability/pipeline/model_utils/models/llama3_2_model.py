
import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

from pipeline.model_utils.model_base import ModelBase

# Llama 3.2 chat templates are based on
# - https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md
# <|begin_of_text|> is automatically added by the tokenizer

LLAMA3_SYSTEM_PROMPT = """Cutting Knowledge Date: December 2023
Today Date: 02 May 2025"""

LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM = """<|start_header_id|>system<|end_header_id|>

{system_prompt}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

LLAMA3_REFUSAL_TOKS = [40, 2170, 358, 1666] # 'I', 'As', ' I', ' As'

def format_instruction_llama3_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True
):
    if system is not None:
        if system == "default":
            system = LLAMA3_SYSTEM_PROMPT
        formatted_instruction = LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system_prompt=system)
    else:
        formatted_instruction = LLAMA3_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_llama3_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True
):
    if outputs is not None:
        prompts = [
            format_instruction_llama3_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_llama3_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

class Llama3_2Model(ModelBase):

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

    def _get_tokenize_instructions_fn(self, system=None):
        return functools.partial(tokenize_instructions_llama3_chat, tokenizer=self.tokenizer, system=system, include_trailing_whitespace=True)
    def _get_model_block_modules(self):
        return self.model.model.layers
