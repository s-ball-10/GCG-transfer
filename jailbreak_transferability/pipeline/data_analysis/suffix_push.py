import os
import sys
sys.path.append(os.path.abspath(os.curdir))
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from jailbreak_transferability.pipeline.utils.utils import get_json_data, MODEL_LAYER_MAPPING
from jailbreak_transferability.pipeline.utils.settings import Settings

SETTINGS = Settings()

acti_path = SETTINGS.models["general"]["activations_path"]

DATA_PATH = "jailbreak_transferability/jailbreak_transferability_data/dataset"


def generate_success_dict(model_name, suffix_model=None):
    #model_name = model_name.split("--")[-1]
    model_name = model_name.lower()
    if suffix_model is None:
        print("Generating success dict for base model transfer")
        if "vicuna" not in model_name and "llama-2-" not in model_name:
            data = get_json_data(f"{DATA_PATH}/multiple_seed_results/{model_name}_multiple_seed_results_transfer.json")
        else:
            data = get_json_data(f"{DATA_PATH}/{model_name}_transfer_data.json")
    else:
        print("Generating success dict for cross model transfer")
        data = get_json_data(f"{DATA_PATH}/{suffix_model}_to_{model_name}_cross_model_transfer_generations.json")
    
    success_dict = {}

    for item in data:
        prompt_index = item["prompt_id"]
        suffix_index = item["suffix_id"]
        jailbreak_judge = item["jailbroken"]

        # Ensure the prompt_index key exists
        if prompt_index not in success_dict:
            success_dict[prompt_index] = {}

        # Assign the jailbreak_judge value
        if jailbreak_judge:
            success_dict[prompt_index][suffix_index] = 1
        else:
            success_dict[prompt_index][suffix_index] = 0

    return success_dict

def get_suffix_push(model_name, suffix_model=None):
    results = []
    short_model_name = model_name.split("--")[-1].lower()
    
    # Load harmful question activations and harmfulness direction
    harmful_activations = torch.load(f'{acti_path}/{short_model_name}_s_prompt_activations.pt', map_location="cuda")[MODEL_LAYER_MAPPING[short_model_name]]
    harmfulness_direction = torch.load(f'{acti_path}/refusal_directions/refusal_direction_layer_{MODEL_LAYER_MAPPING[short_model_name]}_{short_model_name}.pt')

    # Calculate baseline scores
    activation_tensor = torch.stack(harmful_activations)
    baseline_scores = torch.matmul(activation_tensor, harmfulness_direction.unsqueeze(0).transpose(0, 1)).squeeze()

    # Load suffix x prompt activations
    if suffix_model is None:
        all_suffix_activations = torch.load(f"{acti_path}/{short_model_name}_s_jailbreak_activations.pt", map_location="cuda")  
        short_model_name_suffix = None
    else:
        short_model_name_suffix = suffix_model.split("--")[-1].lower()
        all_suffix_activations = torch.load(f"{acti_path}/{short_model_name_suffix}_{short_model_name}_s_jailbreak_activations.pt", map_location="cuda")
        print(len(all_suffix_activations.keys()))

    success_dict = generate_success_dict(short_model_name, short_model_name_suffix)

    # For each harmful question
    for prompt_idx, base_acti in enumerate(harmful_activations):
        # Calculate baseline projections
        baseline_proj = (torch.dot(base_acti, harmfulness_direction) / 
                        torch.dot(harmfulness_direction, harmfulness_direction)) * harmfulness_direction
        baseline_orthogonal = base_acti - baseline_proj
        
        # For each suffix and its layers
        for suffix_id, layer_dict in all_suffix_activations.items():
            # Get activations for this layer
    
            if MODEL_LAYER_MAPPING[short_model_name] not in layer_dict:
                    print("PROBLEM")
                    continue


            activations = layer_dict[MODEL_LAYER_MAPPING[short_model_name]]
        
            
            # Get the activation for this prompt + suffix combination
            suffix_acti = activations[prompt_idx].to(dtype=base_acti.dtype)  # Assuming activations are ordered by prompt index
            
            # Calculate measurements
            suffix_score = torch.dot(suffix_acti, harmfulness_direction)
            refusal_push = baseline_scores[prompt_idx] - suffix_score
            
            suffix_proj = (torch.dot(suffix_acti, harmfulness_direction) / 
                          torch.dot(harmfulness_direction, harmfulness_direction)) * harmfulness_direction
            suffix_orthogonal = suffix_acti - suffix_proj
            
            orthogonal_shift = torch.norm(suffix_orthogonal - baseline_orthogonal).item()
            
            # Store results
            results.append({
                "prompt_index": prompt_idx,
                "suffix_index": suffix_id,
                "refusal_connectivity": baseline_scores[prompt_idx].item(),
                "suffix_score": suffix_score.item(),
                "suffix_push": refusal_push.item(),
                "orthogonal_shift": orthogonal_shift,
                "effective_ratio": (refusal_push / orthogonal_shift).item() if orthogonal_shift > 0 else 0,
                "jailbreak_success": success_dict[prompt_idx][suffix_id]
            })
    if suffix_model is None: 
        save_path = f"{DATA_PATH}/{short_model_name}_layer_{MODEL_LAYER_MAPPING[short_model_name]}_suffix_push_refusal_connect_combined_multidim.json"
    else: 
        save_path = f"{DATA_PATH}/{short_model_name_suffix}_{short_model_name}_layer_{MODEL_LAYER_MAPPING[short_model_name]}_suffix_push_refusal_connect_combined_multidim.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    
    return results



def visualize_jailbreak_effects_aggregated_by_suffix(jailbreak_results, model_name, suffix_model=None):
    """
    Visualize how different suffixes affect the refusal direction, aggregated across multiple harmful prompts.
    """
    # Convert results to DataFrame
    results_df = pd.DataFrame(jailbreak_results)
    short_model_name = model_name.split("--")[-1]
    
    # Aggregation by suffix 
    suffix_summary = results_df.groupby('suffix_index').agg({
        'suffix_push': 'mean',
        'orthogonal_shift': 'mean',
        'jailbreak_success': 'mean',
    }).reset_index()
    
    # Set up the plot
    plt.figure(figsize=(12,4)) #(12, 8)
    sns.set_style("whitegrid")
    
    # Scatter plot 
    scatter = plt.scatter(
        suffix_summary['suffix_push'], 
        suffix_summary['orthogonal_shift'],
        c=suffix_summary['jailbreak_success'],
        s=100,
        alpha=0.7,
        cmap='RdYlGn',
        edgecolors='black',
        vmin=0, vmax=1
    )
    
    # Colorbar and annotations 
    cbar = plt.colorbar(scatter)
    cbar.set_label('Attack success rate')
    
    '''
    # Annotations remain the same
    for i, row in suffix_summary.iterrows():
        suffix_text = str(row['suffix_index'])
        plt.annotate(
            suffix_text,
            (row['suffix_push'], row['orthogonal_shift']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    '''
    
    # Reference lines and labels 
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Average suffix push (higher = stronger push away from refusal)', fontsize=14) 
    plt.ylabel('Average orthogonal shift \n(changes unrelated to refusal)', fontsize=14) 
    #plt.title('Jailbreak suffix effects on model representations (averaged across harmful prompts)', fontsize=14)
    plt.axhline(y=suffix_summary['orthogonal_shift'].median(), color='gray', linestyle='--', alpha=0.5)
    
    plt.text(
        suffix_summary['suffix_push'].max() * 0.8, 
        suffix_summary['orthogonal_shift'].min() * 1.1,
        "Efficient jailbreaks\n(high push, low noise)",
        fontsize=12,
        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7)
    )
    
    # Move the "Noisy Jailbreaks" text to the top y-axis position
    y_max = suffix_summary['orthogonal_shift'].max() * 1.009
    plt.text(
        suffix_summary['suffix_push'].max() * 0.8, 
        y_max,
        "Noisy jailbreaks\n(high push, high noise)",
        fontsize=12, 
        ha='center', va='top', bbox=dict(facecolor='white', alpha=0.7)
    )
    
    '''
    # Move analysis text to top-left corner
    top_pushers = suffix_summary.nlargest(3, 'suffix_push')
    top_success = suffix_summary.nlargest(3, 'jailbreak_success')
    
    analysis_text = (
        f"Top 3 by refusal push:\n" + 
        "\n".join([f"{i+1}. Suffix {row['suffix_index']} (Push: {row['suffix_push']:.2f}, Success: {row['jailbreak_success']:.0%})" 
                  for i, (_, row) in enumerate(top_pushers.iterrows())]) +
        "\n\nTop 3 by Success Rate:\n" +
        "\n".join([f"{i+1}. Suffix {row['suffix_index']} (Success: {row['jailbreak_success']:.0%}, Push: {row['suffix_push']:.2f})" 
                  for i, (_, row) in enumerate(top_success.iterrows())])
    )
    
    # Updated position to top-left corner
    #plt.figtext(0.02, 0.98, analysis_text, 
    plt.figtext(0.1, 0.92, analysis_text, 
                bbox=dict(facecolor='white', alpha=0.7), 
                fontsize=9,
                verticalalignment='top')  # Align top of text with position
    '''
    
    # Stats text at top-right
    stats_text = (
        f"Summary statistics:\n"
        f"Avg success rate: {suffix_summary['jailbreak_success'].mean():.0%}\n"
        f"Correlation (push vs success): {suffix_summary['suffix_push'].corr(suffix_summary['jailbreak_success']):.2f}\n"
        f"Correlation (orthogonal vs success): {suffix_summary['orthogonal_shift'].corr(suffix_summary['jailbreak_success']):.2f}"
    )
    
     
    plt.figtext(0.1, 0.92, stats_text,
                bbox=dict(facecolor='white', alpha=0.5),
                verticalalignment='top',
                fontsize=12) 
    
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    
    # Save figure
    os.makedirs("results/jailbreak_embedding_analysis", exist_ok=True)
    if suffix_model is None:
        save_path = f"results/jailbreak_embedding_analysis/suffix_push_aggregated_{short_model_name}.png"
    else:
        short_model_name_suffix = suffix_model.split("--")[-1].lower()
        save_path = f"results/jailbreak_embedding_analysis/suffix_push_aggregated_{short_model_name_suffix}_{short_model_name}.png"
    plt.savefig(save_path, dpi=300)
    




if __name__ == "__main__":

    model_names = ["qwen2.5-3b-instruct", "vicuna-13b-v1.5", "llama-2-7b-chat-hf","llama-3.2-1b-instruct" ]
    for model_name in model_names:
        suffix_model=None
        jailbreak_results = get_suffix_push(model_name, suffix_model)
        visualize_jailbreak_effects_aggregated_by_suffix(jailbreak_results, model_name, suffix_model)

    suffixes_models = ["qwen2.5-3b-instruct", "llama-3.2-1b-instruct"]
    base_models = ["llama-3.2-1b-instruct", "qwen2.5-3b-instruct"]

    for suffix_model, base_model in zip(suffixes_models, base_models):
        jailbreak_results = get_suffix_push(base_model, suffix_model)
        visualize_jailbreak_effects_aggregated_by_suffix(jailbreak_results, base_model, suffix_model)




