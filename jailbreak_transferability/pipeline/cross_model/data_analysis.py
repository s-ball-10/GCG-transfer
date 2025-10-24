import argparse
import os 

import pandas as pd
import matplotlib.pyplot as plt

from pipeline.utils import utils
from pipeline.config import Config

from matplotlib.colors import ListedColormap

def parse_args():
    parser = argparse.ArgumentParser(description='Set up data sets for cross-model analysis.')
    parser.add_argument('--source_model_path', type=str, required=True, help='Path to the source model')
    parser.add_argument('--target_model_path', type=str, required=True, help='Path to the target model')
    parser.add_argument('--dataset_parent_dir', type=str, required=False, help='The parent directory of the dataset. If not provided, it will be assumed that the directory is in the root folder.')
    return parser.parse_args()

def plot_success_matrix(success_matrix_df, source_cfg, target_cfg):
    save_dir = os.path.join(source_cfg.FIGURES, f'{source_cfg.model_alias}_to_{target_cfg.model_alias}')
    file_name = "cross_model_transfer_success_matrix.png"
    save_path = os.path.join(save_dir, file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, ax = plt.subplots(figsize=(25, 25))
    cmap = ListedColormap(['white', '#4C72B0'])
    ax.imshow(success_matrix_df, cmap=cmap, interpolation='none')
    # ax.set_title(f"Jailbreak Success Matrix for Transfer from {source_cfg.model_name()} to {target_cfg.model_name()}")
    ax.set_xticks(range(len(success_matrix_df.columns)))
    # ax.set_xticklabels(success_matrix_df.columns)
    ax.set_xticklabels([])
    ax.set_yticks(range(len(success_matrix_df.index)))
    # ax.set_yticklabels(success_matrix_df.index)
    ax.set_yticklabels([])
    ax.set_xlabel('Suffix', fontsize=150)
    ax.set_ylabel('Prompt', fontsize=150)
    plt.savefig(save_path)
    plt.close()

def data_analysis(source_cfg, target_cfg):
    save_dir = os.path.join(target_cfg.cross_model_transfer_generations_dir(), f'{source_cfg.model_alias}_to_{target_cfg.model_alias}')
    file_name = "cross_model_transfer_generations.json"
    save_path = os.path.join(save_dir, file_name)
    transfer_df = pd.read_json(save_path)

    previously_refused_indices = utils.get_previously_refused_indices(target_cfg)
    transfer_df = transfer_df[transfer_df['prompt_id'].isin(previously_refused_indices)]

    success_matrix_df = utils.get_jailbreak_success_matrix_df(transfer_df)
    plot_success_matrix(success_matrix_df, source_cfg, target_cfg)

def main():
    args = parse_args()
    source_cfg = Config(model_path=args.source_model_path, dataset_parent_dir=args.dataset_parent_dir)
    target_cfg = Config(model_path=args.target_model_path, dataset_parent_dir=args.dataset_parent_dir)
    
    print("Source model path", args.source_model_path)
    print("Target model path", args.target_model_path)

    data_analysis(source_cfg, target_cfg)

if __name__ == "__main__":
    main()