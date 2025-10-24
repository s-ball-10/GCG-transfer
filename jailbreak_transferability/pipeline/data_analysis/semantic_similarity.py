import os
import sys
sys.path.append(os.path.abspath(os.curdir))

import pandas as pd
import torch
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

from utils.settings import Settings
from jailbreak_transferability.pipeline.utils.utils import MODEL_LAYER_MAPPING, get_json_data


SETTINGS = Settings()

acti_path = SETTINGS.models["general"]["activations_path"]

DATA_PATH = "jailbreak_transferability/jailbreak_transferability_data/dataset"

def get_independent_prompt_activations_from_embedding_model(model_name="all-mpnet-base-v2"):
    """
    Get embeddings of prompts from an independent sentence embedding model.
    
    Parameters:
        model_name (str): Name of the sentence transformer model to use.
                          Options include:
                          - "all-MiniLM-L6-v2" (fast, compact)
                          - "all-mpnet-base-v2" (higher quality)
        
    Returns:
        list: List of embeddings, where each item corresponds to a prompt
    """
    # Install sentence-transformers if not already installed
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Installing sentence-transformers...")
        import subprocess
        subprocess.check_call(["pip", "install", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer
    
    
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Get prompts
    print("Loading prompt data...")
    data = get_json_data(f"{DATA_PATH}/vicuna-13b-v1.5.json")["jailbreaks"]
    questions = [item['goal'] for item in data]
    print(questions[:4])
    
    print(f"Found {len(questions)} prompts")
    
    # Generate embeddings in batches to avoid memory issues
    print("Generating embeddings...")
    batch_size = 32
    embeddings = []
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings)
        print(f"Processed {min(i+batch_size, len(questions))}/{len(questions)} prompts")
    
    # Combine batches
    all_embeddings = torch.cat(embeddings)
    
    # Convert to list of tensors
    embeddings_list = [all_embeddings[i].detach().cpu() for i in range(len(questions))]
    
    print(f"Generated embeddings with dimension {all_embeddings.shape[1]}")
    
    # Save embeddings
    save_path=f"{acti_path}/{model_name}_s_prompt_embeddings.pt"
    
    torch.save(embeddings_list, save_path)
    print(f"Saved embeddings to {save_path}")
    
    return embeddings_list

def create_suffix_lookup_dict():
    """
    Creates a dictionary where keys are prompt_ids (0-99) and values are lists
    of corresponding suffix_ids for each prompt.
    
    Returns:
        dict: A dictionary mapping prompt_ids to lists of suffix_ids
    """
    suffix_lookup = {}
    
    for prompt_id in range(100):  # 0-99
        start_suffix_id = prompt_id * 100
        suffix_ids = list(range(start_suffix_id, start_suffix_id + 100))
        suffix_lookup[prompt_id] = suffix_ids
    
    return suffix_lookup

def get_semantic_sim(model_name, independent_activations=False, one_dimensional=False):
    """
    Analyze the relationship between semantic similarity and jailbreak transferability.
    
    Parameters:
        model_name (str): Name of the model to analyze
        one_dimensional (bool): If True, use the dataset with one suffix per prompt
                               where suffix_id = prompt_id for the original prompt
    
    Returns:
        pd.DataFrame: Results of the analysis
    """
    if one_dimensional:
        # Load the one-dimensional dataset where each prompt has one suffix
        df = pd.read_json(f"{DATA_PATH}/{model_name}_transfer_data.json")
        print(f"Loaded one-dimensional dataset with {len(df)} entries")
    else:
        # Load multiple_seeds dataset
        df = pd.read_json(f"{DATA_PATH}/multiple_seed_results/{model_name}_multiple_seed_results_transfer.json")
        print(f"Loaded multi-dimensional dataset with {len(df)} entries")

    # Load prompt activations
    if independent_activations==True:
        actis = torch.load(f"{acti_path}/all-mpnet-base-v2_s_prompt_embeddings.pt")
    else:
        actis = torch.load(f"{acti_path}/{model_name}_s_prompt_activations.pt")[MODEL_LAYER_MAPPING[model_name]]
    
    # Create suffix lookup dictionary based on dataset type
    if one_dimensional:
        # In one-dimensional case, each prompt has only one suffix with suffix_id = prompt_id
        suffix_lookup = {prompt_id: [prompt_id] for prompt_id in range(len(actis))}
        print(suffix_lookup)

    else:
        # In multi-dimensional case, each prompt has 100 suffixes
        suffix_lookup = create_suffix_lookup_dict()
    
    # Create dictionary to store results
    results = {}
    
    for i in range(len(actis)):
        for j in range(len(actis)):
            if i == j:  # Skip comparing prompt with itself
                print("Skipping pair")
                continue
                
            prompt_1_id = i
            prompt_2_id = j
            prompt_1_suffixes = suffix_lookup[prompt_1_id]
            prompt_2_suffixes = suffix_lookup[prompt_2_id]
            
            # Calculate cosine similarity between prompt activations
            prompt_1_acti = actis[prompt_1_id]
            prompt_2_acti = actis[prompt_2_id]
            
            # Calculate cosine similarity directly
            cosine_sim = torch.nn.functional.cosine_similarity(
                prompt_1_acti.unsqueeze(0), 
                prompt_2_acti.unsqueeze(0)
            ).item()
            
            # Filter data for prompt_1 with prompt_2's suffixes
            prompt_1_with_prompt_2_suffixes = df[
                (df['prompt_id'] == prompt_1_id) & 
                (df['suffix_id'].isin(prompt_2_suffixes))
            ]
            
            # Filter data for prompt_2 with prompt_1's suffixes
            prompt_2_with_prompt_1_suffixes = df[
                (df['prompt_id'] == prompt_2_id) & 
                (df['suffix_id'].isin(prompt_1_suffixes))
            ]
            
            # Check if we have data for both directions
            if len(prompt_1_with_prompt_2_suffixes) == 0 or len(prompt_2_with_prompt_1_suffixes) == 0:
                # Skip this pair if data is missing for either direction
                print(f"Skipping prompt pair ({prompt_1_id}, {prompt_2_id}) due to missing data")
                continue
            
            # Calculate jailbreak success rates
            p1_jailbroken_by_p2_suffixes = prompt_1_with_prompt_2_suffixes['jailbroken'].mean()
            p2_jailbroken_by_p1_suffixes = prompt_2_with_prompt_1_suffixes['jailbroken'].mean()
            
            # Count successful jailbreaks
            p1_jailbroken_count = prompt_1_with_prompt_2_suffixes['jailbroken'].sum()
            p2_jailbroken_count = prompt_2_with_prompt_1_suffixes['jailbroken'].sum()
            
            # Store results
            results[(prompt_1_id, prompt_2_id)] = {
                'cosine_similarity': cosine_sim,
                'p1_jailbroken_by_p2_suffixes_rate': p1_jailbroken_by_p2_suffixes,
                'p2_jailbroken_by_p1_suffixes_rate': p2_jailbroken_by_p1_suffixes,
                'p1_jailbroken_count': p1_jailbroken_count,
                'p2_jailbroken_count': p2_jailbroken_count,
                'total_p1_with_p2_suffixes': len(prompt_1_with_prompt_2_suffixes),
                'total_p2_with_p1_suffixes': len(prompt_2_with_prompt_1_suffixes)
            }
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Check if we have any results
    if results_df.empty:
        print(f"No valid prompt pairs found for analysis in the {model_name} dataset")
        return pd.DataFrame()  # Return empty DataFrame
    
    results_df.index = pd.MultiIndex.from_tuples(results_df.index, names=['prompt_1_id', 'prompt_2_id'])
    results_df = results_df.reset_index()
    
    # Add average jailbreak rate for each prompt pair
    results_df['avg_jailbreak_rate'] = (results_df['p1_jailbroken_by_p2_suffixes_rate'] + 
                                       results_df['p2_jailbroken_by_p1_suffixes_rate']) / 2
    print(results_df)
    # Calculate correlation between similarity and jailbreak success
    correlation = results_df['cosine_similarity'].corr(results_df['avg_jailbreak_rate'])
    
    print(f"Correlation between semantic similarity and jailbreak transfer success: {correlation}")
    
    return results_df

def create_compact_significance_table_standardized_coefficients(models, independent_activations_options=[False, True], dimensional_options=[False, True]):
    """
    Generate a publication-ready compact table using star notation for significance,
    including standardized coefficients for better comparison across models.
    """
    results = []
    
    for model in models:
        for ind_act in independent_activations_options:
            for one_dim in dimensional_options:
                try:
                    # Get data and run analysis
                    results_df = get_semantic_sim(model, ind_act, one_dim)
                    if results_df.empty:
                        continue
                    
                    # Calculate standardized versions of the variables
                    # Standardize the predictor (cosine_similarity)
                    results_df['cosine_similarity_std'] = (results_df['cosine_similarity'] - results_df['cosine_similarity'].mean()) / results_df['cosine_similarity'].std()
                    
                    # Process based on model type
                    if one_dim:
                        # Standardize the outcome before categorization (for reference)
                        results_df['avg_jailbreak_rate_std'] = (results_df['avg_jailbreak_rate'] - results_df['avg_jailbreak_rate'].mean()) / results_df['avg_jailbreak_rate'].std()
                        
                        # Ordinal logistic regression setup
                        results_df['success_category'] = pd.cut(
                            results_df['avg_jailbreak_rate'], 
                            bins=[-0.001, 0.001, 0.6, 1.001],
                            labels=[0, 1, 2]
                        ).astype(int)
                        
                        # Run model with standardized predictor
                        ordinal_model = OrderedModel(
                            results_df['success_category'], 
                            results_df[['cosine_similarity_std']], 
                            distr='logit'
                        )
                        model_res = ordinal_model.fit(method='bfgs', disp=False)
                        
                        # Extract key statistics
                        coef = model_res.params['cosine_similarity_std']  
                        p_value = model_res.pvalues['cosine_similarity_std']
                        r_squared = model_res.prsquared
                        
                        # Also run unstandardized model for raw coefficient 
                        unstd_model = OrderedModel(
                            results_df['success_category'], 
                            results_df[['cosine_similarity']], 
                            distr='logit'
                        )
                        unstd_res = unstd_model.fit(method='bfgs', disp=False)
                        raw_coef = unstd_res.params['cosine_similarity']
                        
                        # Calculate odds ratio as effect size (from standardized coefficient)
                        odds_ratio = np.exp(coef)
                        effect_size = odds_ratio
                        effect_type = "OR"
                        r_squared_type = "MZ-R²"
                        
                    else:
                        # Standardize the outcome
                        results_df['avg_jailbreak_rate_std'] = (results_df['avg_jailbreak_rate'] - results_df['avg_jailbreak_rate'].mean()) / results_df['avg_jailbreak_rate'].std()
                        
                        # Linear regression with standardized variables
                        X_std = sm.add_constant(results_df['cosine_similarity_std'])
                        y_std = results_df['avg_jailbreak_rate_std']
                        model_res_std = sm.OLS(y_std, X_std).fit()
                        
                        # Also run unstandardized model for raw coefficient
                        X = sm.add_constant(results_df['cosine_similarity'])
                        y = results_df['avg_jailbreak_rate']
                        model_res = sm.OLS(y, X).fit()
                        
                        # Extract key statistics
                        coef = model_res_std.params[1]  # This is now the standardized beta
                        p_value = model_res_std.pvalues[1]  # p-value is the same for both models
                        r_squared = model_res.rsquared  # R² is the same for both models
                        raw_coef = model_res.params[1]  # Raw unstandardized coefficient
                        
                    
                    # Add significance stars
                    sig_stars = ""
                    if p_value < 0.001:
                        sig_stars = "***"
                    elif p_value < 0.01:
                        sig_stars = "**"
                    elif p_value < 0.05:
                        sig_stars = "*"
                    
                    # Format coefficients with stars
                    std_coef_formatted = f"{coef:.2f}{sig_stars}"
                    raw_coef_formatted = f"{raw_coef:.2f}{sig_stars}"
                    
                    # Create result entry
                    result = {
                        'Model': model,
                        'Embed': 'Indep.' if ind_act else 'Model',
                        'Dim': '1D' if one_dim else 'Multi',
                        'Raw Coef': raw_coef_formatted,
                        'Std Coef': std_coef_formatted,
                        'R²': f"{r_squared_type}={r_squared:.2f}"
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error analyzing {model} (independent={ind_act}, one_dim={one_dim}): {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add footnote about significance levels
    sig_note = "* p<0.05, ** p<0.01, *** p<0.001"
    
    # Format for LaTeX table
    latex_table = results_df.to_latex(index=False, escape=False)
    latex_table += f"\n\\multicolumn{{{len(results_df.columns)}}}{{l}}{{\\footnotesize {sig_note}}}"
    
    # Save results
    with open("jailbreak_transferability/pipeline/results/semantic_similarity/semantic_similarity_significance_table_standardized.tex", "w") as f:
        f.write(latex_table)
    
    return results_df, latex_table


if __name__ == "__main__":

    get_independent_prompt_activations_from_embedding_model(model_name="all-mpnet-base-v2")
    models_to_analyze = ["qwen2.5-3b-instruct", "vicuna-13b-v1.5", "llama-3.2-1b-instruct", "llama-2-7b-chat-hf"] 
    create_compact_significance_table_standardized_coefficients(models_to_analyze)




