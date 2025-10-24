import argparse
import pandas as pd
import os

from tqdm import tqdm

from pipeline.config import Config
from pipeline.submodules.jailbreak_judge import Llama3JailbreakJudge

def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Parse arguments.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--dataset_parent_dir', type=str, required=False, help='The parent directory of the dataset. If not provided, it will be assumed that the directory is in the root folder.')
    parser.add_argument('--num_gpus', type=int, required=False, help='The number of GPUs available')
    parser.add_argument('--multi_seed', action=argparse.BooleanOptionalAction)
    parser.add_argument('--chunk_id', required=False, type=int, default=-1, help='Chunk ID to process in multi-seed generation')
    parser.add_argument('--no_suffix_completions', action=argparse.BooleanOptionalAction, help='Whether to generate no suffix completions or not')
    return parser.parse_args()

def evaluate_generations(jailbreak_judge, data_path, batch_size=100, chunk_id=None):
    if chunk_id is not None:
        print("Chunk ID", chunk_id)
        dir, file = os.path.split(data_path)
        data_path = os.path.join(dir, f'{chunk_id}_{file}')
    
    generations_df = pd.read_json(data_path)
    n = len(generations_df)

    if 'jailbroken' not in generations_df.columns:
        generations_df['jailbroken'] = None
        generations_df.to_json(data_path, orient='records', indent=4)

    start = generations_df['jailbroken'].notna().sum()
    print(f"Resuming at index {start}/{n}")

    for batch_start in tqdm(range(start, n, batch_size)):
        batch_end = min(batch_start + batch_size, n)
        prompts = generations_df['prompt'].iloc[batch_start:batch_end].to_list()
        responses = generations_df['response'].iloc[batch_start:batch_end].to_list()
        classifications = jailbreak_judge.classify_responses(prompts, responses)

        for i, classification in enumerate(classifications):
            generations_df.loc[batch_start + i, 'jailbroken'] = classification

        generations_df.to_json(data_path, orient='records', indent=4)

if __name__ == "__main__":
    args = parse_arguments()
    cfg = Config(model_path=args.model_path, dataset_parent_dir=args.dataset_parent_dir)
    jailbreak_judge = Llama3JailbreakJudge(num_gpus=args.num_gpus)
    
    if args.no_suffix_completions:
        print("Evaluating no suffix completions")
        evaluate_generations(jailbreak_judge, data_path=cfg.no_suffix_generations_path(), chunk_id=None)
    elif args.multi_seed:
        print("Evaluating multi-seed generations")
        # Evaluate multi-seed generations
        # evaluate_generations(jailbreak_judge, data_path=cfg.multi_seed_generations_no_transfer_path(), chunk_id=None)
        evaluate_generations(jailbreak_judge, data_path=cfg.multi_seed_generations_transfer_path(), chunk_id=args.chunk_id)
    # Cross prompt transfer generations
    else:
        print("Evaluating cross prompt transfer generations")
        # Evaluate cross prompt transfer generations
        evaluate_generations(jailbreak_judge, data_path=cfg.cross_prompt_transfer_generations_path(), chunk_id=None)
