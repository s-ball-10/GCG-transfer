#!/bin/bash

export HF_HOME=/data/hf_cache
model_id="Qwen/Qwen2.5-1.5B-Instruct"
# model_id="meta-llama/Llama-3.2-1B-Instruct"
# model_id="google/gemma-2b-it"
num_steps=1
end_seed=99
index=0

python gcg.py \
    --model_id $model_id \
    --model_name $(basename $model_id) \
    --num_steps $num_steps \
    --index $index \
    --end_seed $end_seed
