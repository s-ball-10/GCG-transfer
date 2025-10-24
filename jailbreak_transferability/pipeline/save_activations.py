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
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--dataset_parent_dir', type=str, required=False, help='The parent directory of the dataset. If not provided, it will be assumed that the directory is in the root folder.')
    parser.add_argument('--multi_seed', action=argparse.BooleanOptionalAction)
    parser.add_argument('--num_chunks', type=int, default=1, help='Number of chunks to split the activations into.')
    parser.add_argument('--s', action=argparse.BooleanOptionalAction, help='Whether to save activations for s or not.')
    parser.add_argument('--prompts', action=argparse.BooleanOptionalAction, help='Whether to save activations for prompts or not.')
    parser.add_argument('--jailbreak', action=argparse.BooleanOptionalAction, help='Whether to save activations for jailbreak or not.')
    return parser.parse_args()

def save_activations(model, data_path_func, input_column, save_path_func):
    data_df = pd.read_json(data_path_func())
    inputs = data_df[input_column].to_list()
    activations = get_activations(model, inputs, model.model_block_modules)

    save_path = save_path_func()
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(activations, save_path)

def save_cosine_similarity_with_refusal(model, refusal_direction, num_layers, data_dir_func, input_column, save_path_func, batch_size=100):
    data_df = utils.concat_json_files_in_dir(data_dir_func())
    inputs = data_df[input_column].to_list()
    
    total_n_nans = 0
    cosine_similarities = torch.empty(len(inputs), num_layers)
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_inputs = inputs[i:i + batch_size]
        acts = get_activations(model, batch_inputs, model.model_block_modules)

        n_nan = torch.isnan(acts).sum().item()
        total_n_nans += n_nan

        cosine_similarities[i:i + batch_size] = torch.nn.functional.cosine_similarity(acts, refusal_direction, dim=-1)

    print(f"Number of NaN values in activations: {total_n_nans}")

    save_path = save_path_func()
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(cosine_similarities, save_path)

def split_and_save_activations(model, data_path_func, input_column, save_dir_func, num_chunks=1, batch_size=100):
    data_df = pd.read_json(data_path_func())
    inputs = data_df[input_column].to_list()

    activations_list = []
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_inputs = inputs[i:i + batch_size]
        acts = get_activations(model, batch_inputs, model.model_block_modules)
        activations_list.append(acts)
    
    activations_list = [t.half() for t in activations_list]
    activations = torch.cat(activations_list, dim=0)

    n_nan = torch.isnan(activations).sum().item()
    print(f"Number of NaN values in activations: {n_nan}")

    save_dir = save_dir_func()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    chunks = torch.chunk(activations, num_chunks, dim=0)
    for i in tqdm(range(num_chunks)):
        save_path = os.path.join(save_dir, f"jailbreak_activations_chunk_{i}.pt")
        torch.save(chunks[i].clone(), save_path)

def save_jailbreak_activations_for_s(model, data_dir_func, input_column, save_dir_func, layers, batch_size=100, num_chunks=1):
    print("Saving jailbreak activations for s...")

    if os.path.isdir(data_dir_func()):
        data_df = utils.concat_json_files_in_dir(data_dir_func())
    else:
        data_df = pd.read_json(data_dir_func())
        
    inputs = data_df[input_column].to_list()

    total_n_nans = 0

    activations_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size)):
            batch_inputs = inputs[i:i + batch_size]
            acts = get_activations(model, batch_inputs, model.model_block_modules)
            acts = acts[:, layers, :].detach().half().cpu()

            n_nan = torch.isnan(acts).sum().item()
            total_n_nans += n_nan

            activations_list.append(acts)
    
    print(f"Number of NaN values in activations: {total_n_nans}")

    activations = torch.cat(activations_list, dim=0)

    new_save_dir = cfg.multi_seed_jailbreak_activations_transfer_dir()
    if not os.path.exists(new_save_dir):
        os.makedirs(new_save_dir)
    torch.save(activations, os.path.join(new_save_dir, 'jailbreak_activations.pt'))
    print(f"Saved activations to {os.path.join(new_save_dir, 'jailbreak_activations.pt')}")


    num_examples, num_layers, d_model = activations.shape
    num_prompts = cfg.num_prompts
    num_suffixes = num_examples // num_prompts

    activations = activations.view(num_prompts, num_suffixes, num_layers, d_model)

    save_dir = os.path.join(cfg.s_activations_dir(), 'jailbreak_activations')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

        save_path = os.path.join(save_dir, f'{chunk_id}_{cfg.model_alias}_s_jailbreak_activations.pt')
        torch.save(activations_dict, save_path)

def save_prompt_activations_for_s(model, layers, cfg):
    data_df = pd.read_json(cfg.prompts_path())
    inputs = data_df['prompt'].to_list()

    activations = get_activations(model, inputs, model.model_block_modules)[:, layers, :]
    num_prompts, _, _ = activations.shape

    n_nan = torch.isnan(activations).sum().item()
    print(f"Number of NaN values in activations: {n_nan}")

    activations_dict = {
        layer_id: [
            activations[prompt_id, i, :].cpu()
            for prompt_id in range(num_prompts)
        ]
        for i, layer_id in enumerate(layers)
    }

    save_dir = os.path.join(cfg.s_activations_dir(), 'prompt_activations')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f'{cfg.model_alias}_s_prompt_activations.pt')
    torch.save(activations_dict, save_path)

    print(f"Saved activations to {save_path}")

def save_prompt_activations(model, cfg):
    save_activations(model, cfg.prompts_path, 'prompt', cfg.prompt_activations_path)

def save_suffix_activations(model, cfg):
    save_activations(model, cfg.suffixes_path, 'suffix', cfg.suffix_activations_path)

def save_jailbreak_activations(model, cfg, num_chunks=1):
    split_and_save_activations(model, cfg.cross_prompt_transfer_generations_path, 'jailbreak', cfg.jailbreak_activations_dir, num_chunks=num_chunks)

def save_multi_seed_jailbreak_activations(model, cfg, num_chunks=1):
    split_and_save_activations(model, cfg.multi_seed_generations_transfer_path, 'jailbreak', cfg.multi_seed_jailbreak_activations_dir, num_chunks=num_chunks)

if __name__ == "__main__":
    args = parse_arguments()
    cfg = Config(model_path=args.model_path, dataset_parent_dir=args.dataset_parent_dir)
    model = construct_model_base(cfg.model_path)

    #TODO: Add support for exctracting suffixes
    # save_suffix_activations(model=model, cfg=cfg)

    refusal_direction = utils.get_refusal_directions(cfg.arditi_et_al_refusal_direction_dir())[0]
    refusal_direction_layer = cfg.arditi_et_al_refusal_direction_layer(0)
    num_layers = len(model.model_block_modules)
    layers = [refusal_direction_layer, num_layers - 1]

    if args.prompts:
        if args.s:
            print("Saving prompt activations for s...")
            save_prompt_activations_for_s(model=model, layers=layers, cfg=cfg)
        else:
            print("Saving prompt activations...")
            save_prompt_activations(model=model, cfg=cfg)

    if args.jailbreak:
        if args.multi_seed:
            if args.s:
                print("Saving multi-seed jailbreak activations for s...")
                save_jailbreak_activations_for_s(
                    model=model, 
                    data_dir_func=cfg.multi_seed_generations_transfer_dir, 
                    input_column='jailbreak', 
                    save_dir_func=cfg.s_activations_dir, 
                    layers=layers,
                    num_chunks=args.num_chunks)
            else:
                print("Saving multi-seed jailbreak activations...")
                save_cosine_similarity_with_refusal(
                    model=model, 
                    refusal_direction=refusal_direction, 
                    num_layers=num_layers, 
                    data_dir_func=cfg.multi_seed_generations_transfer_dir, 
                    input_column='jailbreak', 
                    save_path_func=cfg.cosine_similarity_with_refusal_path)
        else:
            if args.s:
                print("Saving jailbreak activations for s...")
                save_jailbreak_activations_for_s(
                    model=model, 
                    data_dir_func=cfg.cross_prompt_transfer_generations_path, 
                    input_column='jailbreak', 
                    save_dir_func=cfg.s_activations_dir, 
                    layers=list(range(num_layers)))

            else:
                print("Saving jailbreak activations...")
                save_jailbreak_activations(model=model, cfg=cfg, num_chunks=args.num_chunks)