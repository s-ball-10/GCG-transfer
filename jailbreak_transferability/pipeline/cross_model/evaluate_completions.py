import argparse
import os 

import pandas as pd

from tqdm import tqdm

from pipeline.config import Config
from pipeline.submodules.jailbreak_judge import Llama3JailbreakJudge

def parse_args():
    parser = argparse.ArgumentParser(description='Set up data sets for cross-model analysis.')
    parser.add_argument('--source_model_path', type=str, required=True, help='Path to the source model')
    parser.add_argument('--target_model_path', type=str, required=True, help='Path to the target model')
    parser.add_argument('--dataset_parent_dir', type=str, required=False, help='The parent directory of the dataset. If not provided, it will be assumed that the directory is in the root folder.')
    parser.add_argument('--chunk_id', type=int, required=True, help='Chunk ID to process')
    parser.add_argument('--num_gpus', type=int, required=False, help='The number of GPUs available')
    return parser.parse_args()

def evaluate_completions(jailbreak_judge, source_cfg, target_cfg, chunk_id, batch_size=100):
    print("Chunk ID", chunk_id)

    save_dir = os.path.join(target_cfg.cross_model_transfer_generations_dir(), f'{source_cfg.model_alias}_to_{target_cfg.model_alias}')
    file_name = f'{chunk_id}_cross_model_transfer_generations.json'
    save_path = os.path.join(save_dir, file_name)

    chunk_df = pd.read_json(save_path)
    n = len(chunk_df)

    if 'jailbroken' not in chunk_df.columns:
        chunk_df['jailbroken'] = None
        chunk_df.to_json(save_path, orient='records', indent=4)

    # Start from the first index where response is not None
    start = chunk_df['jailbroken'].notna().sum()
    print(f"Resuming at index {start}/{n}")

    for batch_start in tqdm(range(start, n, batch_size)):
        batch_end = min(batch_start + batch_size, n)
        prompts = chunk_df['prompt'].iloc[batch_start:batch_end].to_list()
        responses = chunk_df['response'].iloc[batch_start:batch_end].to_list()
        classifications = jailbreak_judge.classify_responses(prompts, responses)

        for i, classification in enumerate(classifications):
            chunk_df.loc[batch_start + i, 'jailbroken'] = classification

        chunk_df.to_json(save_path, orient='records', indent=4)

def main():
    args = parse_args()
    source_cfg = Config(model_path=args.source_model_path, dataset_parent_dir=args.dataset_parent_dir)
    target_cfg = Config(model_path=args.target_model_path, dataset_parent_dir=args.dataset_parent_dir)
    
    print("Source model path", args.source_model_path)
    print("Target model path", args.target_model_path)

    jailbreak_judge = Llama3JailbreakJudge(num_gpus=args.num_gpus)
    evaluate_completions(jailbreak_judge, source_cfg, target_cfg, args.chunk_id)

if __name__ == "__main__":
    main()