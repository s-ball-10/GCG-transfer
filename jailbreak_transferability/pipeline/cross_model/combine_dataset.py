import argparse
import os 

import pandas as pd

from pipeline.utils import utils
from pipeline.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Set up data sets for cross-model analysis.')
    parser.add_argument('--source_model_path', type=str, required=True, help='Path to the source model')
    parser.add_argument('--target_model_path', type=str, required=True, help='Path to the target model')
    parser.add_argument('--dataset_parent_dir', type=str, required=False, help='The parent directory of the dataset. If not provided, it will be assumed that the directory is in the root folder.')
    parser.add_argument('--num_chunks', type=int, default=1, help='Number of chunks to split the dataset into')
    return parser.parse_args()

def combine_dataset(source_cfg, target_cfg, num_chunks=1):
    save_dir = os.path.join(target_cfg.cross_model_transfer_generations_dir(), f'{source_cfg.model_alias}_to_{target_cfg.model_alias}')
    df = utils.concat_json_files_in_dir(save_dir)

    for filename in os.listdir(save_dir):
        full_path = os.path.join(save_dir, filename)
        if os.path.isfile(full_path):
            os.remove(full_path)

    if num_chunks == 1:
        df.to_json(os.path.join(save_dir, "cross_model_transfer_generations.json"), orient='records', indent=4)
    else:
        num_examples_per_chunk = len(df) // num_chunks
        for chunk_id in range(num_chunks):
            start_index = chunk_id * num_examples_per_chunk
            end_index = (chunk_id + 1) * num_examples_per_chunk if chunk_id != num_chunks - 1 else len(df)
            chunk_df = df[start_index:end_index]
            chunk_path = os.path.join(save_dir, f"{chunk_id}_cross_model_transfer_generations.json")
            chunk_df.to_json(chunk_path, orient='records', indent=4)

def main():
    args = parse_args()
    source_cfg = Config(model_path=args.source_model_path, dataset_parent_dir=args.dataset_parent_dir)
    target_cfg = Config(model_path=args.target_model_path, dataset_parent_dir=args.dataset_parent_dir)
    
    print("Source model path", args.source_model_path)
    print("Target model path", args.target_model_path)

    combine_dataset(source_cfg, target_cfg, args.num_chunks)

if __name__ == "__main__":
    main()