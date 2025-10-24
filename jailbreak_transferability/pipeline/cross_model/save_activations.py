import argparse
import torch
import os

import pandas as pd
from tqdm import tqdm

from pipeline.utils import utils
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_activations import get_activations

def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Parse arguments.")
    parser.add_argument('--source_model_path', type=str, required=True, help='Path to the source model')
    parser.add_argument('--target_model_path', type=str, required=True, help='Path to the target model')
    parser.add_argument('--dataset_parent_dir', type=str, required=False, help='The parent directory of the dataset. If not provided, it will be assumed that the directory is in the root folder.')
    parser.add_argument('--s', action=argparse.BooleanOptionalAction, help='Whether to save activations for s or not.')
    return parser.parse_args()

# def save_activations(model, data_path_func, input_column, save_path_func):
#     data_df = pd.read_json(data_path_func())
#     inputs = data_df[input_column].to_list()
#     activations = get_activations(model, inputs, model.model_block_modules)

#     save_path = save_path_func()
#     save_dir = os.path.dirname(save_path)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     torch.save(activations, save_path)

def save_jailbreak_activations_for_s(target_model, source_cfg, target_cfg, layers, batch_size=100, num_chunks=1):
    data_dir = os.path.join(target_cfg.cross_model_transfer_generations_dir(), f'{source_cfg.model_alias}_to_{target_cfg.model_alias}')
    data_path = os.path.join(data_dir, 'cross_model_transfer_generations.json')
    data_df = pd.read_json(data_path)
    inputs = data_df['jailbreak'].to_list()

    total_n_nans = 0

    activations_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size)):
            batch_inputs = inputs[i:i + batch_size]
            acts = get_activations(target_model, batch_inputs, target_model.model_block_modules)
            acts = acts[:, layers, :].detach().half().cpu()

            n_nan = torch.isnan(acts).sum().item()
            total_n_nans += n_nan

            activations_list.append(acts)
    
    print(f"Number of NaN values in activations: {total_n_nans}")

    activations = torch.cat(activations_list, dim=0)

    save_dir = os.path.join(target_cfg.activations_dir(), f'{source_cfg.model_alias}_to_{target_cfg.model_alias}')

    new_save_dir = os.path.join(save_dir, 'jailbreak_activations')
    if not os.path.exists(new_save_dir):
        os.makedirs(new_save_dir)
    torch.save(activations, os.path.join(new_save_dir, 'jailbreak_activations.pt'))
    print(f"Saved activations to {os.path.join(new_save_dir, 'jailbreak_activations.pt')}")

    num_examples, num_layers, d_model = activations.shape
    num_prompts = target_cfg.num_prompts
    num_suffixes = num_examples // num_prompts

    activations = activations.view(num_prompts, num_suffixes, num_layers, d_model)

    s_save_dir = os.path.join(save_dir, 's_activations', 'jailbreak_activations')
    if not os.path.exists(s_save_dir):
        os.makedirs(s_save_dir)

    num_suffixes_per_chunk = num_suffixes // num_chunks
    for chunk_id in tqdm(range(num_chunks)):
        chunk_start = chunk_id * num_suffixes_per_chunk
        chunk_end = (chunk_id + 1) * num_suffixes_per_chunk if chunk_id != num_chunks - 1 else num_suffixes

        activations_dict = {
            suffix_id: {
                layer_id: [
                    activations[prompt_id, suffix_id, i, :].clone()
                    for prompt_id in range(num_prompts)
                ]
                for i, layer_id in enumerate(layers)
            }
            for suffix_id in range(chunk_start, chunk_end)
        }

        torch.save(activations_dict, os.path.join(s_save_dir, 'jailbreak_activations.pt'))

if __name__ == "__main__":
    args = parse_arguments()
    source_cfg = Config(model_path=args.source_model_path, dataset_parent_dir=args.dataset_parent_dir)
    target_cfg = Config(model_path=args.target_model_path, dataset_parent_dir=args.dataset_parent_dir)
    target_model = construct_model_base(target_cfg.model_path)

    num_layers = len(target_model.model_block_modules)
    layers = list(range(num_layers))

    if args.s:
        print("Saving multi-seed jailbreak activations for s...")
        save_jailbreak_activations_for_s(
            target_model=target_model, 
            source_cfg=source_cfg,
            target_cfg=target_cfg,
            layers=layers)