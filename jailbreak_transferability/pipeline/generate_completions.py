import argparse
import pandas as pd
import os

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base

def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Parse arguments.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--dataset_parent_dir', type=str, required=False, help='The parent directory of the dataset. If not provided, it will be assumed that the directory is in the root folder.')
    parser.add_argument('--multi_seed', action=argparse.BooleanOptionalAction)
    parser.add_argument('--num_chunks', type=int, default=1, help='Number of chunks to split the dataset into for multi-seed generation')
    parser.add_argument('--chunk_id', type=int, default=0, help='Chunk ID to process in multi-seed generation')
    parser.add_argument('--no_transfer', action=argparse.BooleanOptionalAction, help='Whether to generate no transfer completions or not')
    parser.add_argument('--no_suffix_completions', action=argparse.BooleanOptionalAction, help='Whether to generate no suffix completions or not')
    return parser.parse_args()

def generate_completions_without_suffixes(model, cfg):
    prompts_df = pd.read_json(cfg.prompts_path())
    inputs = prompts_df['prompt'].to_list()
    completions = model.generate_completions(inputs, max_new_tokens=200)
    
    no_suffix_df = prompts_df[['prompt_id', 'prompt']].copy()
    no_suffix_df['response'] = completions

    no_suffix_df.to_json(cfg.no_suffix_generations_path(), orient='records', indent=4)

def generate_cross_prompt_transfer_completions(model, cfg):
    prompts_df = pd.read_json(cfg.prompts_path()).drop(columns=['category'])
    suffixes_df = pd.read_json(cfg.suffixes_path())

    cross_transfer_df = prompts_df.merge(suffixes_df, how='cross')

    if 'old_suffix_id' in cross_transfer_df.columns:
        cross_transfer_df = cross_transfer_df[['prompt_id', 'suffix_id', 'old_suffix_id','prompt', 'suffix']]
    else:
        cross_transfer_df = cross_transfer_df[['prompt_id', 'suffix_id', 'prompt', 'suffix']]

    cross_transfer_df['jailbreak'] = cross_transfer_df['prompt'] + cross_transfer_df['suffix']

    inputs = cross_transfer_df['jailbreak'].to_list()
    completions = model.generate_completions(inputs, max_new_tokens=200)
    cross_transfer_df['response'] = completions

    cross_transfer_df.to_json(cfg.cross_prompt_transfer_generations_path(), orient='records', indent=4)

def generate_multi_seed_completions_transfer(model, path, num_chunks=1, chunk_id=0):
    print("Num chunks and chunk ID", num_chunks, chunk_id)
    assert chunk_id < num_chunks, "chunk_id must be less than num_chunks"

    df = pd.read_json(path)

    num_examples_per_chunk = len(df) // num_chunks
    start_index = chunk_id * num_examples_per_chunk
    end_index = (chunk_id + 1) * num_examples_per_chunk if chunk_id != num_chunks - 1 else len(df)
    df = df[start_index:end_index]

    inputs = df['jailbreak'].to_list()
    completions = model.generate_completions(inputs, max_new_tokens=200)
    df['response'] = completions

    dir, file = os.path.split(path)
    save_path = os.path.join(dir, f'{chunk_id}_{file}')
    df.to_json(save_path, orient='records', indent=4)

def generate_multi_seed_completions_no_transfer(model, path):
    df = pd.read_json(path)

    inputs = df['jailbreak'].to_list()
    completions = model.generate_completions(inputs, max_new_tokens=200)
    df['response'] = completions

    df.to_json(path, orient='records', indent=4)

if __name__ == "__main__":
    args = parse_arguments()
    cfg = Config(model_path=args.model_path, dataset_parent_dir=args.dataset_parent_dir)
    print("Model path", args.model_path)
    model = construct_model_base(cfg.model_path)

    if args.no_suffix_completions:
        print("Generating no suffix completions")
        generate_completions_without_suffixes(model=model, cfg=cfg)

    if args.multi_seed:
        if args.no_transfer:
            print("Generating multi-seed completions without transfer")
            generate_multi_seed_completions_no_transfer(model=model, path=cfg.multi_seed_generations_no_transfer_path())
        else:
            print("Generating multi-seed completions with transfer")
            generate_multi_seed_completions_transfer(model=model, path=cfg.multi_seed_generations_transfer_path(), num_chunks=args.num_chunks, chunk_id=args.chunk_id)
    else:
        print("Generating cross prompt transfer completions")
        generate_cross_prompt_transfer_completions(model=model, cfg=cfg)

