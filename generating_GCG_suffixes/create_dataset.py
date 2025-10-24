import os
import json
import argparse
import pandas as pd
import numpy as np

def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Parse arguments.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    return parser.parse_args()

"""Aggregate all the results from multiple seeds into a single JSON file."""
def create_no_transfer_dataset(model_path, model_alias):
    results = []
    suffix_index = 0
    for root, dir, files in os.walk(f'results/{model_alias}'):
        dir.sort()
        files.sort()
        if 'results.json' in files:
            file_path = os.path.join(root, 'results.json')
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                record = {
                    "prompt_id": data["index"],
                    "suffix_id": suffix_index,
                    "seed": data["seed"],
                    "prompt": data["goal"],
                    "suffix": data["best_string"],
                    "jailbreak": data["goal"] + data["best_string"],
                    "loss": data["best_loss"],
                    "num_steps": data["strings"].index(data["best_string"]) + 1
                }

                suffix_index += 1
                results.append(record)

    with open(f'{model_alias.lower()}_multiple_seed_results_no_transfer.json', 'w') as f:
        json.dump(results, f, indent=4)

def create_transfer_dataset(df):
    
    prompt_meta = (
        df[["prompt_id", "prompt"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    suffix_meta = (
        df.drop(columns=["prompt_id", "prompt", "jailbreak"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    df_combos = prompt_meta.merge(suffix_meta, how="cross")

    df_combos["jailbreak"] = df_combos["prompt"] + df_combos["suffix"]

    original_pairs = (
        df[["prompt_id", "suffix_id"]]
        .drop_duplicates()
        .assign(transfer=False)
    )

    df_combos = (
        df_combos
        .merge(original_pairs, on=["prompt_id", "suffix_id"], how="left")
    )

    df_combos["transfer"] = df_combos["transfer"].isna()
    return df_combos[['prompt_id', 'suffix_id', 'seed', 'prompt', 'suffix', 'jailbreak', 'loss', 'num_steps', 'transfer']]

def test_create_transfer_dataset():
    input_df = pd.DataFrame({
        "prompt_id": [0, 0, 1, 1],
        "suffix_id": [0, 1, 2, 3],
        "seed": [0, 1, 0, 1],
        "prompt": ["prompt0", "prompt0", "prompt1", "prompt1"],
        "suffix": ["suffix0", "suffix1", "suffix2", "suffix3"],
        "jailbreak": ["prompt0suffix0", "prompt0suffix1", "prompt1suffix2", "prompt1suffix3"],
        "loss": [0.5, 0.6, 0.7, 0.8],
        "num_steps": [10, 20, 30, 40]
    })

    expected_output = pd.DataFrame({
        "prompt_id": [0, 0, 0, 0, 1, 1, 1, 1],
        "suffix_id": [0, 1, 2, 3, 0, 1, 2, 3],
        "seed": [0, 1, 0, 1, 0, 1, 0, 1],
        "prompt": ["prompt0", "prompt0", "prompt0", "prompt0", "prompt1", "prompt1", "prompt1", "prompt1"],
        "suffix": ["suffix0", "suffix1", "suffix2", "suffix3", "suffix0", "suffix1", "suffix2", "suffix3"],
        "jailbreak": ["prompt0suffix0", "prompt0suffix1", "prompt0suffix2", "prompt0suffix3", "prompt1suffix0", "prompt1suffix1", "prompt1suffix2", "prompt1suffix3"],
        "loss": [0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8],
        "num_steps": [10, 20, 30, 40, 10, 20, 30, 40],
        "transfer": [False, False, True, True, True, True, False, False]
    })

    output_df = create_transfer_dataset(input_df)

    pd.testing.assert_frame_equal(output_df, expected_output)

if __name__ == "__main__":
    args = parse_arguments()
    model_path = args.model_path
    model_alias = os.path.basename(model_path)

    create_no_transfer_dataset(model_path, model_alias)

    test_create_transfer_dataset()

    df = pd.read_json(f'{model_alias.lower()}_multiple_seed_results_no_transfer.json')
    df_combos = create_transfer_dataset(df)
    df_combos.to_json(f'{model_alias.lower()}_multiple_seed_results_transfer.json', orient='records', indent=4)
