import json
import pandas as pd

# Change model files and number of rephrases here
NUMBER_OF_REPHRASES = 10
MODEL_NAME ="qwen2.5-3b-instruct" #"llama-3.2-1b-instruct"  

# Load doc
with open(f"jailbreak_transferability/pipeline/data_analysis/rephrase_experiments/results/{MODEL_NAME}_prompt_rephrasings.json", "r") as f:
    data = json.load(f)

# collapse
df = pd.DataFrame(data=data)

# Group by rephrased_prompt_id and get mean of jailbroken column
grouped_df = df.groupby("rephrased_prompt_id")["jailbroken"].mean()

# collect suffix IDs
suffix_ids = []
for item in data:
    suffix_id = item["old_suffix_id"] 
    if suffix_id not in suffix_ids:
        suffix_ids.append(suffix_id)

# original prompt IDs
ori_prompt_ids = []
for item in data:
    prompt_id = item["original_prompt_id"] 
    if prompt_id not in ori_prompt_ids:
        ori_prompt_ids.append(prompt_id)

# original data
original_data = pd.read_json(f"jbb_transfer_code/jailbreak_transferability/jailbreak_transferability_data/dataset/multiple_seed_results/{MODEL_NAME}_multiple_seed_results_transfer.json")

# Filter original data for the relevant prompt_ids and suffix_ids
filtered_original = original_data[
    (original_data["prompt_id"].isin(ori_prompt_ids)) & 
    (original_data["suffix_id"].isin(suffix_ids))
]

# Group by prompt_id and get mean of jailbroken column
original_grouped = filtered_original.groupby("prompt_id")["jailbroken"].mean()


# Load the paraphrase dataset with dot products 
paraphrase_data = pd.read_json(f"jailbreak_transferability/pipeline/data_analysis/rephrase_experiments/results/{MODEL_NAME}/experiment_results_unaligned_model_vicuna-13b-v1.5.json") 

print("\n=== DOT PRODUCT vs ASR ANALYSIS ===")

# Create a mapping from rephrased prompt text to ASR (using the grouped means we calculated earlier)
text_to_asr = {}
for rephrased_id, asr in grouped_df.items():
    # Get the prompt text for this rephrased_prompt_id
    prompt_text = df[df["rephrased_prompt_id"] == rephrased_id]["prompt"].iloc[0]
    text_to_asr[prompt_text.strip()] = asr

# Analyze each original prompt and its paraphrases
results = []
for _, row in paraphrase_data.iterrows():
    prompt_id = row['prompt_id']
    original_prompt = row['prompt']
    ori_dot_product = row['ori_dot_product']
    
    # Get original ASR
    original_asr = original_grouped.get(prompt_id, None)
    
    # Check each paraphrase
    for i in range(NUMBER_OF_REPHRASES):  
        paraphrase_text = row[f'paraphrase_{i}']
        dot_product = row[f'dot_product_{i}']
        
        # Find ASR for this paraphrase by matching text
        paraphrase_asr = None
        for text, asr in text_to_asr.items():
            # Clean and match text
            if text.strip() == paraphrase_text.strip():
                paraphrase_asr = asr
                break
        
        if paraphrase_asr is not None and original_asr is not None:
            results.append({
                'prompt_id': prompt_id,
                'paraphrase_id': i,
                'paraphrase_text': paraphrase_text[:50] + "..." if len(paraphrase_text) > 50 else paraphrase_text,
                'original_asr': original_asr,
                'paraphrase_asr': paraphrase_asr,
                'original_dot_product': ori_dot_product,
                'paraphrase_dot_product': dot_product,
                'asr_change': paraphrase_asr - original_asr,
                'dot_product_change': dot_product - ori_dot_product
            })
        else:
            # Debug: print unmatched paraphrases
            print(f"No ASR match found for prompt_id {prompt_id}, paraphrase {i}: '{paraphrase_text[:50]}...'")

# Convert to DataFrame for analysis
analysis_df = pd.DataFrame(results)


if not analysis_df.empty:
    print(f"\nFound {len(analysis_df)} matched paraphrases")
    print("\nPrompt ID | Para ID | Orig ASR | Para ASR | ASR Change | Orig Dot | Para Dot | Dot Change")
    print("-" * 100)
    
    for _, row in analysis_df.iterrows():
        print(f"{row['prompt_id']:8} | {row['paraphrase_id']:7} | {row['original_asr']:8.4f} | {row['paraphrase_asr']:8.4f} | {row['asr_change']:10.4f} | {row['original_dot_product']:8.1f} | {row['paraphrase_dot_product']:8.1f} | {row['dot_product_change']:10.1f}")
    
    # Correlation analysis with statistical significance
    from scipy import stats
    
    correlation, p_value = stats.pearsonr(analysis_df['asr_change'], analysis_df['dot_product_change'])
    print(f"\nCorrelation between ASR change and dot product change: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Sample size: {len(analysis_df)}")
    
    # Interpret significance
    alpha = 0.05
    if p_value < alpha:
        print(f"✓ Correlation is statistically significant at α = {alpha}")
    else:
        print(f"✗ Correlation is NOT statistically significant at α = {alpha}")
    
    # Calculate confidence interval for correlation
    # Using Fisher's z-transformation for confidence interval
    import numpy as np
    
    if len(analysis_df) > 3:  # Need at least 4 points for CI
        z_score = 0.5 * np.log((1 + correlation) / (1 - correlation))
        se = 1 / np.sqrt(len(analysis_df) - 3)
        ci_lower_z = z_score - 1.96 * se
        ci_upper_z = z_score + 1.96 * se
        ci_lower = (np.exp(2 * ci_lower_z) - 1) / (np.exp(2 * ci_lower_z) + 1)
        ci_upper = (np.exp(2 * ci_upper_z) - 1) / (np.exp(2 * ci_upper_z) + 1)
        print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Power analysis warning
    if len(analysis_df) < 30:
        print(f"\n⚠️  WARNING: Small sample size (n={len(analysis_df)}) may lead to:")
        print("   - Low statistical power")
        print("   - Unreliable correlation estimates")
        print("   - High influence of outliers")
    
    # Effect size interpretation
    abs_corr = abs(correlation)
    if abs_corr >= 0.7:
        effect_size = "large"
    elif abs_corr >= 0.5:
        effect_size = "medium"
    elif abs_corr >= 0.3:
        effect_size = "small"
    else:
        effect_size = "negligible"
    
    print(f"\nEffect size: {effect_size} (|r| = {abs_corr:.4f})")
    
    # Additional robustness checks
    spearman_corr, spearman_p = stats.spearmanr(analysis_df['asr_change'], analysis_df['dot_product_change'])
    print(f"\nSpearman correlation (rank-based): {spearman_corr:.4f}, p = {spearman_p:.4f}")
    
    if abs(correlation - spearman_corr) > 0.2:
        print("⚠️  Large difference between Pearson and Spearman suggests non-linear relationship or outliers")
    
    
    # Check for outliers
    print(f"\n=== OUTLIER ANALYSIS ===")
    print(f"Dot product change range: {analysis_df['dot_product_change'].min():.1f} to {analysis_df['dot_product_change'].max():.1f}")
    print(f"ASR change range: {analysis_df['asr_change'].min():.4f} to {analysis_df['asr_change'].max():.4f}")
    
    # Show extreme cases
    max_dot_change = analysis_df.loc[analysis_df['dot_product_change'].idxmax()]
    min_dot_change = analysis_df.loc[analysis_df['dot_product_change'].idxmin()]
    max_asr_change = analysis_df.loc[analysis_df['asr_change'].idxmax()]
    min_asr_change = analysis_df.loc[analysis_df['asr_change'].idxmin()]
    
    print(f"\nMax dot product change: {max_dot_change['dot_product_change']:.1f}, ASR change: {max_dot_change['asr_change']:.4f}")
    print(f"Min dot product change: {min_dot_change['dot_product_change']:.1f}, ASR change: {min_dot_change['asr_change']:.4f}")
    print(f"Max ASR change: {max_asr_change['asr_change']:.4f}, dot product change: {max_asr_change['dot_product_change']:.1f}")
    print(f"Min ASR change: {min_asr_change['asr_change']:.4f}, dot product change: {min_asr_change['dot_product_change']:.1f}")
    
    # Categorize by dot product change direction
    higher_dot = analysis_df[analysis_df['dot_product_change'] > 0]
    lower_dot = analysis_df[analysis_df['dot_product_change'] < 0]
    
    print(f"\nParaphrases with HIGHER dot product (n={len(higher_dot)}):")
    if not higher_dot.empty:
        print(f"  Average ASR change: {higher_dot['asr_change'].mean():.4f}")
        print(f"  Average dot product change: {higher_dot['dot_product_change'].mean():.1f}")
        print(f"  ASR increased in {len(higher_dot[higher_dot['asr_change'] > 0])} cases")
        print(f"  ASR decreased in {len(higher_dot[higher_dot['asr_change'] < 0])} cases")
    
    print(f"\nParaphrases with LOWER dot product (n={len(lower_dot)}):")
    if not lower_dot.empty:
        print(f"  Average ASR change: {lower_dot['asr_change'].mean():.4f}")
        print(f"  Average dot product change: {lower_dot['dot_product_change'].mean():.1f}")
        print(f"  ASR increased in {len(lower_dot[lower_dot['asr_change'] > 0])} cases")
        print(f"  ASR decreased in {len(lower_dot[lower_dot['asr_change'] < 0])} cases")
    
    # Show some examples
    print(f"\n=== EXAMPLES ===")
    print("Cases where higher dot product led to lower ASR:")
    higher_dot_lower_asr = analysis_df[(analysis_df['dot_product_change'] > 0) & (analysis_df['asr_change'] < 0)]
    for _, row in higher_dot_lower_asr.head(3).iterrows():
        print(f"  Prompt {row['prompt_id']}: Dot +{row['dot_product_change']:.1f}, ASR {row['asr_change']:.4f}")
        print(f"    '{row['paraphrase_text']}'")
    
    print("\nCases where lower dot product led to higher ASR:")
    lower_dot_higher_asr = analysis_df[(analysis_df['dot_product_change'] < 0) & (analysis_df['asr_change'] > 0)]
    for _, row in lower_dot_higher_asr.head(3).iterrows():
        print(f"  Prompt {row['prompt_id']}: Dot {row['dot_product_change']:.1f}, ASR +{row['asr_change']:.4f}")
        print(f"    '{row['paraphrase_text']}'")
    
else:
    print("No matching paraphrases found. Check text matching logic.")
    print("Available paraphrase texts (first 3):")
    for i, row in paraphrase_data.head(3).iterrows():
        for j in range(NUMBER_OF_REPHRASES):
            print(f"  {row['prompt_id']}-{j}: '{row[f'paraphrase_{j}'][:50]}...'")
    
    print("\nAvailable ASR texts (first 3):")
    for i, (text, asr) in enumerate(list(text_to_asr.items())[:3]):
        print(f"  {i}: '{text[:50]}...' -> ASR: {asr:.4f}")



