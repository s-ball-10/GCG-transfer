import os
import sys
sys.path.append(os.path.abspath(os.curdir))

import torch
import json
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

from jailbreak_transferability.pipeline.utils.utils import MODEL_LAYER_MAPPING, get_prompt_only_dataset
from jailbreak_transferability.pipeline.utils.settings import Settings



SETTINGS = Settings()
acti_path = SETTINGS.models["general"]["activations_path"]

# Load the activations
def get_similarity_scores(model_name):
    harmful_activations = torch.load(f'{acti_path}/{model_name}_s_prompt_activations.pt', map_location=torch.device('cpu'))[MODEL_LAYER_MAPPING[model_name]] 
    harmfulness_direction = torch.load(f'{acti_path}/refusal_directions/refusal_direction_layer_{MODEL_LAYER_MAPPING[model_name]}_{model_name}.pt', map_location=torch.device('cpu')).to(dtype=harmful_activations[0].dtype)

    # calculate cosine similarity between harmful questions and harmfulness direction
    # Concatenate the list of activations into a single tensor
    activation_tensor = torch.stack(list(harmful_activations))
    activation_tensor_cpu = activation_tensor.cpu()
    harmfulness_direction_cpu = harmfulness_direction.cpu()
    similarity_cosine = cosine_similarity(activation_tensor_cpu, harmfulness_direction_cpu.unsqueeze(0)) #
    similarity_dot = torch.matmul(activation_tensor_cpu, harmfulness_direction_cpu.unsqueeze(0).transpose(0, 1))
    
    # Get dataset that stores prompt, category, and similarity

    data = get_prompt_only_dataset(model_name)

    store_in = []

    for i,item in enumerate(data): 
        
        n_item = {"prompt": item["prompt"],  
                "prompt_index": item["prompt_id"],
                "category": item["category"], 
                "cos_sim_to_harm": similarity_cosine[i].item(),
                "dot_product_to_harm": similarity_dot[i].item()}
        store_in.append(n_item)
  
    return store_in

def plot_simple_dispersion_comparison(models_data, model_names, figsize=(12, 4), save_path=None):
    ## fig size before was 12, 6
    """
    Plot KDE distributions with transparent shaded regions showing dispersion.
    
    Parameters:
    -----------
    models_data : list of list of dict
        List where each element contains the data for one model
    model_names : list of str
        List of model names corresponding to each element in models_data
    figsize : tuple
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Generate a color palette with distinct colors for each model
    num_models = len(model_names)
    #palette = sns.color_palette("tab10", num_models) #viridis 
    custom_colors = [
    "#ff7f00",  # orange
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ffff33",  # yellow
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # gray
    "#000000"   # black
    ]
    palette = custom_colors[:num_models] 
        
    # For storing legend elements
    legend_elements = []
    
    # Process each model's data
    for i, (data, model_name) in enumerate(zip(models_data, model_names)):
        # Convert to DataFrame
        df = pd.DataFrame(data)
        color = palette[i]
        
        # Calculate statistics for cosine similarity
        cos_mean = df['cos_sim_to_harm'].mean()
        cos_std = df['cos_sim_to_harm'].std()
        
        # Normalize dot product
        max_abs_dot = max(abs(df['dot_product_to_harm'].max()), abs(df['dot_product_to_harm'].min()))
        df['normalized_dot_product'] = df['dot_product_to_harm'] / max_abs_dot if max_abs_dot > 0 else df['dot_product_to_harm']
        
        # Calculate statistics for normalized dot product
        dot_mean = df['normalized_dot_product'].mean()
        dot_std = df['normalized_dot_product'].std()
        
        # Plot KDE for cosine similarity
        sns.kdeplot(
            df['cos_sim_to_harm'], 
            ax=ax1, 
            color=color,
            alpha=0.8,
            #fill=True,  # Fill under the curve
            linewidth=2,
            label=f"{model_name} (σ={cos_std:.3f})"  # Include std dev in label
        )
        
        # Plot KDE for normalized dot product
        sns.kdeplot(
            df['normalized_dot_product'], 
            ax=ax2, 
            color=color,
            alpha=0.8,
            #fill=True,  # Fill under the curve
            linewidth=2,
            label=f"{model_name} (σ={dot_std:.3f})"  # Include std dev in label
        )
    
    # Add vertical lines at 0 for reference
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Set titles and labels
    #ax1.set_title('Cosine similarity to refusal direction', fontsize=14)
    ax1.set_xlabel('Cosine similarity with refusal direction', fontsize=14) 
    ax1.set_ylabel('Density', fontsize=12)
    
    #ax2.set_title('Normalized dot Product with Harmfulness Direction', fontsize=14)
    ax2.set_xlabel('Normalized dot product with refusal direction', fontsize=14) 
    ax2.set_ylabel('Density', fontsize=12)
    
    # Add figure title highlighting dispersion comparison
    #fig.suptitle('Distribution Spread Comparison Across Models', fontsize=16, y=0.98)
    
    # Add legends to both plots (including standard deviation in the label)
    ax1.legend(fontsize=11) 
    ax2.legend(fontsize=11)
    
    # Grid for better readability
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Optional: Save the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    
    models_data = [get_similarity_scores("qwen2.5-3b-instruct"), get_similarity_scores("vicuna-13b-v1.5"), 
    get_similarity_scores("llama-2-7b-chat-hf"), get_similarity_scores("llama-3.2-1b-instruct")]
    model_names = ["Qwen", "Vicuna", "Llama 2", "Llama 3.2"]
    plot_simple_dispersion_comparison(models_data, model_names, figsize=(12, 4), save_path="./results/harmfulness_activation/cross_model_density_plot.png")
    




