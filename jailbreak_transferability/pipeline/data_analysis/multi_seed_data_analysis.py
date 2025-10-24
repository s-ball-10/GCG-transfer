import argparse
import torch

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from pipeline.utils import utils
from pipeline.config import Config

from pipeline.utils.activation_utils import get_prompt_activations

def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Parse arguments.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--dataset_parent_dir', type=str, required=False, help='The parent directory of the dataset. If not provided, it will be assumed that the directory is in the root folder.')
    return parser.parse_args()

def plot_with_error_bars(layers, mean_values, std_values, color):
    plt.plot(layers, mean_values, label=None, color=color)
    plt.fill_between(layers, mean_values - std_values, mean_values + std_values, alpha=0.2, color=color)

def plot_cosine_similarity_across_layers(cosine_similarities, color, num_layers):
    mean_values = cosine_similarities.mean(dim=0).cpu().detach().numpy()
    std_values = cosine_similarities.std(dim=0).cpu().detach().numpy()
    layers = np.arange(num_layers)

    plot_with_error_bars(layers, mean_values, std_values, color)

def plot_cosine_similarity_with_refusal(cosine_similarities, prompt_cosine_sim_with_refusal, num_layers, cfg, most_successful_suffix_ids, least_successful_suffix_ids):
    plot_cosine_similarity_across_layers(
        prompt_cosine_sim_with_refusal,
        color="blue",
        num_layers=num_layers
    )

    for suffix_id in most_successful_suffix_ids:
        plot_cosine_similarity_across_layers(
            cosine_similarities[:, suffix_id, :], 
            color="green",
            num_layers=num_layers)
    
    for suffix_id in least_successful_suffix_ids:
        plot_cosine_similarity_across_layers(
            cosine_similarities[:, suffix_id, :], 
            color="orange",
            num_layers=num_layers)
        
    plt.xlabel("Layer number", fontsize=24)
    plt.ylabel("Cosine sim w refusal", fontsize=24)
    # plt.title(f"{cfg.model_name().capitalize()} Average Cosine Similarity with Refusal Direction Across Layers (Multi-Seed)", fontsize=10)
    # plt.legend(handles=[
        # plt.Line2D([0], [0], color='blue', label='Harmful Prompt'),
        # plt.Line2D([0], [0], color='green', label='Harmful Prompt + Top 3 Most Successful Suffixes'),
        # plt.Line2D([0], [0], color='orange', label='Harmful Prompt + Top 3 Least Successful Suffixes')
    # ])
    plt.tight_layout()
    plt.savefig(f'figures/{cfg.model_alias}/cosine_similarity_across_layers_suffixes_direction_multi_seed.png')
    plt.close()

def data_analysis(cfg):
    ### No Transfer ###
    no_transfer_multi_seed_df = utils.get_multi_seed_df(cfg, transfer=False)

    no_transfer_success_matrix_df = utils.get_multi_seed_jailbreak_success_matrix_no_transfer_df(no_transfer_multi_seed_df)
    utils.plot_jailbreak_success_matrix(no_transfer_success_matrix_df, cfg, x_label="Seed", y_label="Prompt", multi_seed=True)

    num_prompts, num_seeds = no_transfer_success_matrix_df.shape

    number_of_successful_jailbreaks = no_transfer_success_matrix_df.sum(axis=1).sum(axis=0)
    fraction_of_successful_jailbreaks = number_of_successful_jailbreaks / (num_prompts * num_seeds)
    print(f"Fraction of successful jailbreaks: {fraction_of_successful_jailbreaks:.2}")

    number_of_prompts_with_at_least_one_successful_jailbreak = (no_transfer_success_matrix_df.sum(axis=1) > 0).sum()
    fraction_of_prompts_with_at_least_one_successful_jailbreak = number_of_prompts_with_at_least_one_successful_jailbreak / num_prompts
    print(f"Fraction of prompts with at least one successful jailbreak: {fraction_of_prompts_with_at_least_one_successful_jailbreak:.2}")

    ### Transfer ###
    n = 3

    transfer_multi_seed_df = utils.get_multi_seed_df(cfg, transfer=True)
    transfer_success_matrix_df = utils.get_jailbreak_success_matrix_df(transfer_multi_seed_df)
    num_prompts, num_suffixes = transfer_success_matrix_df.shape

    most_successful_suffixes = transfer_success_matrix_df.sum(axis=0).nlargest(n).index.tolist()
    least_successful_suffixes = transfer_success_matrix_df.sum(axis=0).nsmallest(n).index.tolist()

    print(f"Most successful suffixes: {most_successful_suffixes}")
    print(f"Least successful suffixes: {least_successful_suffixes}")

    # refactor into own function
    consine_sim_with_refusal = torch.load(cfg.cosine_similarity_with_refusal_path(), weights_only=True)
    consine_sim_with_refusal = consine_sim_with_refusal.reshape(cfg.num_prompts, num_suffixes, -1)
    prev_refused_indices = utils.get_previously_refused_indices(cfg).tolist()
    consine_sim_with_refusal = consine_sim_with_refusal[prev_refused_indices, :, :]
    _, _, num_layers = consine_sim_with_refusal.shape

    # Plot the cosine similarity with refusal direction across layers for the most and least successful suffixes
    refusal_direction = utils.get_refusal_directions(cfg.arditi_et_al_refusal_direction_dir())[0].half()
    prompt_activations = get_prompt_activations(cfg)

    prompt_cosine_sim_with_refusal = torch.nn.functional.cosine_similarity(prompt_activations, refusal_direction, dim=-1)
    print(prompt_cosine_sim_with_refusal.shape)

    plot_cosine_similarity_with_refusal(
        consine_sim_with_refusal,
        prompt_cosine_sim_with_refusal,
        num_layers, 
        cfg, 
        most_successful_suffix_ids=most_successful_suffixes, 
        least_successful_suffix_ids=least_successful_suffixes
    )



if __name__ == "__main__":
    args = parse_arguments()
    cfg = Config(model_path=args.model_path, dataset_parent_dir=args.dataset_parent_dir)
    data_analysis(cfg)
