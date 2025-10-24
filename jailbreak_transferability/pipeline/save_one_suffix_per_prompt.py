import argparse
import torch

import numpy as np
import pandas as pd

from pipeline.utils import utils
from pipeline.config import Config

def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Parse arguments.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--dataset_parent_dir', type=str, required=False, help='The parent directory of the dataset. If not provided, it will be assumed that the directory is in the root folder.')
    return parser.parse_args()

def save_one_suffix_per_prompt(cfg):
    no_transfer_df = pd.read_json(cfg.multi_seed_generations_no_transfer_path())
    no_transfer_df_sampled = no_transfer_df.groupby('prompt_id', group_keys=False).sample(n=1, random_state=42)
    no_transfer_df_sampled = no_transfer_df_sampled.reset_index(drop=True)
    df_suffixes = no_transfer_df_sampled[['suffix_id', 'suffix']]
    df_suffixes.rename(columns={'suffix_id': 'old_suffix_id'}, inplace=True)
    df_suffixes['suffix_id'] = df_suffixes.index
    df_suffixes = df_suffixes[['suffix_id', 'old_suffix_id', 'suffix']]
    return df_suffixes

def get_df_subset(df_suffixes, multi_seed_df):
    mapping = df_suffixes.set_index('old_suffix_id')['suffix_id']
    sub = multi_seed_df[multi_seed_df['suffix_id'].isin(mapping.index)].copy()
    sub['new_suffix_id'] = sub['suffix_id'].map(mapping)

    sub.rename(columns={
        'suffix_id':     'old_suffix_id',
        'new_suffix_id': 'suffix_id'
    }, inplace=True)

    cols = [
        'prompt_id',
        'old_suffix_id',  # was multi_seed_df.suffix_id
        'suffix_id',      # from df_suffixes
        'seed','prompt','suffix','jailbreak',
        'loss','num_steps','transfer','response','jailbroken'
    ]
    
    df_subset = sub[cols]

    return df_subset

if __name__ == "__main__":
    args = parse_arguments()
    cfg = Config(model_path=args.model_path, dataset_parent_dir=args.dataset_parent_dir)
    df_suffixes = save_one_suffix_per_prompt(cfg)
    df_suffixes.to_json(cfg.suffixes_path(), orient='records', indent=4)

    multi_seed_dir = cfg.multi_seed_generations_transfer_dir()
    multi_seed_df = utils.concat_json_files_in_dir(multi_seed_dir)

    multi_seed_df = get_df_subset(df_suffixes, multi_seed_df)

    save_path = cfg.cross_prompt_transfer_generations_path()
    multi_seed_df.to_json(save_path, orient='records', indent=4)



