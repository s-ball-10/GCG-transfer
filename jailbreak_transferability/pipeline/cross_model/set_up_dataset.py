import argparse
import os 

import pandas as pd

from pipeline.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Set up data sets for cross-model analysis.')
    parser.add_argument('--source_model_path', type=str, required=True, help='Path to the source model')
    parser.add_argument('--target_model_path', type=str, required=True, help='Path to the target model')
    parser.add_argument('--dataset_parent_dir', type=str, required=False, help='The parent directory of the dataset. If not provided, it will be assumed that the directory is in the root folder.')
    parser.add_argument('--num_chunks', type=int, default=1, help='Number of chunks to split the dataset into')
    return parser.parse_args()

def set_up_dataset(source_cfg, target_cfg, num_chunks=1):
    prompts_df = pd.read_json(source_cfg.prompts_path()).drop(columns=['category'])
    source_suffixes_df = pd.read_json(source_cfg.suffixes_path())
    source_multi_seed_no_transfer_df = pd.read_json(source_cfg.multi_seed_generations_no_transfer_path())
    
    # Add GCG loss and num_steps to source_suffixes_df
    source_suffixes_df = (
        source_suffixes_df
        .join(
            source_multi_seed_no_transfer_df.set_index('suffix_id')[['loss', 'num_steps']],
            on='old_suffix_id'
        )
        [['suffix_id', 'old_suffix_id', 'suffix', 'loss', 'num_steps']]
    )

    # Get all combinations of prompts and suffixes and add 'jailbreak' column
    source_input_df = prompts_df.merge(source_suffixes_df, how='cross')
    source_input_df['jailbreak'] = source_input_df['prompt'] + source_input_df['suffix']
    source_input_df = source_input_df[['prompt_id', 'suffix_id', 'old_suffix_id', 'prompt', 'suffix', 'jailbreak', 'loss', 'num_steps']]

    save_dir = os.path.join(target_cfg.cross_model_transfer_generations_dir(), f'{source_cfg.model_alias}_to_{target_cfg.model_alias}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = "cross_model_transfer_generations.json"

    num_examples_per_chunk = len(source_input_df) // num_chunks
    for chunk_id in range(num_chunks):
        start_index = chunk_id * num_examples_per_chunk
        end_index = (chunk_id + 1) * num_examples_per_chunk if chunk_id != num_chunks - 1 else len(source_input_df)
        chunk_df = source_input_df[start_index:end_index]
        chunk_path = os.path.join(save_dir, f"{chunk_id}_{file_name}")
        chunk_df.to_json(chunk_path, orient='records', indent=4)

def main():
    args = parse_args()
    source_cfg = Config(model_path=args.source_model_path, dataset_parent_dir=args.dataset_parent_dir)
    target_cfg = Config(model_path=args.target_model_path, dataset_parent_dir=args.dataset_parent_dir)
    
    print("Source model path", args.source_model_path)
    print("Target model path", args.target_model_path)

    set_up_dataset(source_cfg, target_cfg, args.num_chunks)

if __name__ == "__main__":
    main()