import os
import sys
sys.path.append(os.path.abspath(os.curdir))

import json
import torch
import math
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

# Open a list of json files and load them into a single list
def load_json_files(file_paths: list[str]):
    if not file_paths:
        raise ValueError('No files provided.')

    with open(file_paths[0], 'r') as f:
        data = json.load(f)

    for file_path in file_paths[1:]:
        with open(file_path, 'r') as f:
            data.extend(json.load(f))

    return data

def get_refusal_directions(refusal_dir):
    refusal_file_paths = [
        os.path.join(refusal_dir, filename)
        for filename in os.listdir(refusal_dir)
        if filename.endswith('.pt') and os.path.isfile(os.path.join(refusal_dir, filename))
    ]
    refusal_directions = [torch.load(file_path, weights_only=True).to('cuda').float() for file_path in refusal_file_paths]
    return refusal_directions

# Get the indices of the prompts that were refused/not jailbroken without a suffix
def get_previously_refused_indices(cfg):
    no_suffix_df = pd.read_json(cfg.no_suffix_generations_path())
    return no_suffix_df.loc[~no_suffix_df['jailbroken'], 'prompt_id']

def get_previously_refused_suffix_indices(cfg):
    suffix_df = pd.read_json(cfg.suffixes_path())
    previously_refused_prompt_indices = get_previously_refused_indices(cfg)
    return suffix_df.loc[previously_refused_prompt_indices, 'suffix_id']

# Filter out entries that were jailbroken even without a suffix
def get_transfer_df(cfg):
    transfer_df = pd.read_json(cfg.cross_prompt_transfer_generations_path())
    refused_prompt_indices = get_previously_refused_indices(cfg)
    refused_suffix_indices = get_previously_refused_suffix_indices(cfg)
    return transfer_df[
        transfer_df['prompt_id'].isin(refused_prompt_indices) 
        & 
        transfer_df['suffix_id'].isin(refused_suffix_indices)
    ]

def get_multi_seed_df(cfg, transfer=True):
    transfer_df = concat_json_files_in_dir(cfg.multi_seed_generations_transfer_dir())
    if not transfer:
        transfer_df = transfer_df[transfer_df['transfer'] == False]
    refused_prompt_indices = get_previously_refused_indices(cfg)
    return transfer_df[
        transfer_df['prompt_id'].isin(refused_prompt_indices)
    ]

# Create a matrix of size num_prompts x num_suffixes where an entry is True if and only if the prompt was jailbroken with the suffix
# Returns a dataframe
def get_jailbreak_success_matrix_df(transfer_df):
    success_matrix = transfer_df.pivot(index='prompt_id', columns='suffix_id', values='jailbroken')
    return success_matrix.astype(int)

def plot_jailbreak_success_matrix(success_matrix, cfg, x_label, y_label, multi_seed=False):
    if multi_seed:
        save_path = cfg.multi_seed_jailbreak_success_matrix_figure_path()
    else:
        save_path = cfg.jailbreak_success_matrix_figure_path()
    fig, ax = plt.subplots(figsize=(25, 25))
    cmap = ListedColormap(['white', '#4C72B0'])
    ax.imshow(success_matrix, cmap=cmap, interpolation='none')
    # ax.set_title(f"Jailbreak Success Matrix for {cfg.model_name()}")
    ax.set_xticks(range(len(success_matrix.columns)))
    # ax.set_xticklabels(success_matrix.columns)
    ax.set_xticklabels([])
    ax.set_yticks(range(len(success_matrix.index)))
    # ax.set_yticklabels(success_matrix.index)
    ax.set_yticklabels([])
    ax.set_xlabel(x_label, fontsize=150)
    ax.set_ylabel(y_label, fontsize=150)
    ax.xaxis.set_label_coords(0.5, -0.035)  # Adjust the vertical position of the xlabel
    ax.yaxis.set_label_coords(-0.035, 0.5)  # Adjust the horizontal position of the ylabel
    plt.savefig(save_path)
    plt.close()

def get_multi_seed_jailbreak_success_matrix_no_transfer_df(df):
    success_matrix = df.pivot(index='prompt_id', columns='seed', values='jailbroken')
    return success_matrix.astype(int)

# Returns a num_prompts x num_layers x d_model tensor
def get_prompt_activations(cfg):
    prompt_activations = torch.load(cfg.prompt_activations_path(), weights_only=True)
    # idx = torch.as_tensor(get_previously_refused_indices(cfg), dtype=torch.long)
    # return prompt_activations[idx]
    return prompt_activations

# Load the activation chunks and concatenate them
def concat_jailbreak_activations(cfg):
    chunk_files = sorted(os.listdir(cfg.jailbreak_activations_dir()))

    chunk_list = []
    for chunk_file in chunk_files:
        print(f"Loading {chunk_file}")
        chunk_path = os.path.join(cfg.jailbreak_activations_dir(), chunk_file)
        chunk = torch.load(chunk_path, weights_only=True, map_location='cpu')
        chunk_list.append(chunk)
    
    activations = torch.cat(chunk_list, dim=0)

    n_nan = torch.isnan(activations).sum().item()
    print(f"Number of NaN values in activations after concat jailbreak activations: {n_nan}")

    return activations

# Returns a num_prompts x num_suffixes x num_layers x d_model tensor
def get_jailbreak_activations(cfg):
    jailbreak_activations = concat_jailbreak_activations(cfg)
    n = int(math.sqrt(jailbreak_activations.shape[0]))
    d = jailbreak_activations.shape[-1]
    jailbreak_activations = jailbreak_activations.view(n, n, -1, d)
    refused_ids = torch.as_tensor(get_previously_refused_indices(cfg), dtype=torch.long)
    # return jailbreak_activations[refused_ids][:, refused_ids, :, :]
    n_nan = torch.isnan(jailbreak_activations).sum().item()
    print(f"Number of NaN values in activations after reshaping jailbreak activations: {n_nan}")
    return jailbreak_activations

# Assumes that the json files all start with a number
# and are sorted in the order they should be concatenated
# Returns a dataframe
def concat_json_files_in_dir(dir_path):
    files = os.listdir(dir_path)
    files.sort(key=lambda x: int(x.split('_', 1)[0]))  # Sort files by the number at the start of the filename

    all_data = []
    for file in files:
        file_path = os.path.join(dir_path, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_data.extend(data)
    
    return pd.DataFrame(all_data)


MODEL_LAYER_MAPPING = {
    "llama-3.2-1b-instruct": 8,
    "qwen2.5-3b-instruct": 27,
    "vicuna-13b-v1.5": 17, 
    "llama-2-7b-chat-hf": 14,
    }

def get_json_data(data_path): 
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def get_prompt_only_dataset(model_name):
    if "vicuna" in model_name or "llama-2" in model_name:
        data_path = f"data/{model_name}_transfer_data.json"
    else:
        data_path = f"data/{model_name}_multiple_seed_results_transfer.json"
    with open(data_path, 'r') as f:
        data = json.load(f)

    prompt_category_dict = {}

    # Load category data:
    with open("data/vicuna-13b-v1.5.json", 'r') as f:
        data_category = json.load(f)

        for item in data_category["jailbreaks"]:
            if item["index"] not in prompt_category_dict:
                prompt_category_dict[item["index"]] = item["category"]
            

    seen_prompt_ids = set()
    filtered_data = []

    for item in data:
        prompt_id = item['prompt_id']
        
        if prompt_id not in seen_prompt_ids:
            if prompt_id in prompt_category_dict:
                item["category"] = prompt_category_dict[prompt_id]
            else: 
                print("Mismatch of prompt text for following prompts:")
                print("\n")
                print(prompt_id)
                print(item["prompt"])
            filtered_data.append(item)
            seen_prompt_ids.add(prompt_id)

    # Now filtered_data contains only the first instance of each unique prompt_id
    return filtered_data
