import os
import sys
sys.path.append(os.path.abspath(os.curdir))
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from jailbreak_transferability.pipeline.utils.utils import MODEL_LAYER_MAPPING

DATA_PATH = "jailbreak_transferability/jailbreak_transferability_data/dataset"


def perform_logistic_regression(jailbreak_results, X_values:list, interaction:list, model_name, suffix_model=None, visualize_interaction=True):
    """
    Perform logistic regression with multiple interaction terms and visualize the results.
    
    Parameters:
    -----------
    jailbreak_results : list or dict
        The jailbreak results data
    X_values : list
        List of feature column names to use in the model
    interaction : list
        List of features to create interactions for. Features at positions i and i+1 
        will form interaction pairs (when i is even).
    visualize_interaction : bool, default=True
        Whether to visualize interaction effects
    """
    # Convert results list to DataFrame
    df = pd.DataFrame(jailbreak_results)

    # Calculate point-biserial correlations
    print("Point-Biserial Correlations:")
    for x in X_values:
        corr = stats.pointbiserialr(df['jailbreak_success'], df[x])
        print(f"{x} and jailbreak_success: {corr[0]:.4f} (p={corr[1]:.4f})")
    
    # Create features and target
    X = df[X_values]
    y = df['jailbreak_success']

    corr_matrix = X.corr()
    print(f"Correlation matrix: ")
    print(corr_matrix)
    print("\n" + "-"*50 + "\n")

    # Scale if needed: 
    scaler = StandardScaler()
    X[X_values] = scaler.fit_transform(X[X_values])

    # Extract interaction pairs from the interaction list
    interaction_pairs = []
    if len(interaction) > 1:
        # Create interaction terms for consecutive pairs (1&2, 3&4, 5&6, etc.)
        for i in range(0, len(interaction), 2):
            # Check if we have both elements of the pair
            if i+1 < len(interaction):
                # Create interaction between consecutive elements
                feature1 = interaction[i]
                feature2 = interaction[i+1]
                interaction_column = f'interaction_{feature1}_{feature2}'
                X[interaction_column] = X[feature1] * X[feature2]
                interaction_pairs.append((feature1, feature2))

    print("Feature matrix with interactions:")
    print(X.head())
    print("\n" + "-"*50 + "\n")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit logistic regression model (sklearn for metrics)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Print sklearn model coefficients
    print("\nModel Coefficients (sklearn balanced):")
    feature_names = X_train.columns
    for feature, coef in zip(feature_names, model.coef_[0]):
        print(f"{feature}: {coef:.4f}")
    print(f"Intercept: {model.intercept_[0]:.4f}")

    # Use statsmodels for coefficient significance tests
    X_train_sm = sm.add_constant(X_train)
    sm_model = sm.Logit(y_train, X_train_sm).fit()
    
    # Print model performance metrics
    print("Model Performance:")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"AIC/BIC: {sm_model.aic:.4f}, {sm_model.bic:.4f}")
    print(classification_report(y_test, y_pred))
    print("\n" + "-"*50 + "\n")

    # Optional: Print full statsmodels summary
    print("\nDetailed Statistics:")
    print(sm_model.summary())

    if visualize_interaction and interaction_pairs:
        print(f"\nVisualizing {len(interaction_pairs)} interaction pairs...")
        visualize_interactions_compact(sm_model, X, interaction_pairs, model_name, suffix_model)
        
    return sm_model, X, interaction_pairs

def visualize_interactions_compact(sm_model, X, interaction_pairs, model_name, suffix_model=None, font_size=14):
    """
    Visualize multiple interaction effects in a compact format suitable for publication.
    Creates a single figure with probability contour plots for each interaction pair.
    
    Parameters:
    -----------
    sm_model : statsmodels regression model
        The fitted model with interaction terms
    X : pandas DataFrame
        The feature dataframe
    interaction_pairs : list of tuples
        List of (feature1, feature2) tuples representing interactions to visualize
    model_name : str
        Name of the model for file naming
    suffix_model : str, optional
        Additional suffix for file naming
    font_size : int, optional
        Base font size for the figure (default=14)
    """
    # Create directory for results if it doesn't exist
    os.makedirs("results/jailbreak_embedding_analysis", exist_ok=True)
    
    # Set global font sizes
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size + 2,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size - 2,
        'ytick.labelsize': font_size - 2,
        'legend.fontsize': font_size - 2,
        'figure.titlesize': font_size + 4
    })
    
    # Calculate number of rows and columns for the subplot grid
    n_interactions = len(interaction_pairs)
    n_cols = min(3, n_interactions)  # Maximum 3 plots per row
    n_rows = (n_interactions + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Handle the case where there's only one interaction pair
    if n_interactions == 1:
        axes = np.array([axes])
    
    # Flatten axes array for easy iteration
    axes_flat = axes.flatten() if n_interactions > 1 else [axes]
    
    # Visualize each interaction pair
    for idx, (feature1, feature2) in enumerate(interaction_pairs):
        if idx >= len(axes_flat):
            break  # Safety check
            
        ax = axes_flat[idx]
        
        # Create a grid of values for the two features
        x_min, x_max = X[feature1].min() - 0.1, X[feature1].max() + 0.1
        y_min, y_max = X[feature2].min() - 0.1, X[feature2].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # Create prediction grid with all features at their mean values
        grid_data = {}
        
        # First, set all features to their mean values
        for col in X.columns:
            if col.startswith('interaction_'):
                continue  # Skip interaction columns, will be calculated
            grid_data[col] = np.repeat(X[col].mean(), 100*100)
        
        # Then override the two features we visualize
        grid_data[feature1] = xx.ravel()
        grid_data[feature2] = yy.ravel()
        
        # Calculate all required interactions
        for pair in interaction_pairs:
            f1, f2 = pair
            grid_data[f'interaction_{f1}_{f2}'] = grid_data[f1] * grid_data[f2]
            
        # Create grid DataFrame for prediction
        grid = pd.DataFrame(grid_data)
        
        # Ensure constant term is included
        grid_with_const = sm.add_constant(grid, has_constant='add')
        
        # Predict probabilities
        Z = sm_model.predict(grid_with_const).to_numpy().reshape(xx.shape)
        
        # Define nice names mapping
        nice_names = {
            "suffix_push": "Suffix push",
            "refusal_connectivity": "Refusal connec.",
            "orthogonal_shift": "Orthogonal shift"
        }
        
        # Get nice names for the features
        f1_nice = nice_names.get(feature1, feature1)
        f2_nice = nice_names.get(feature2, feature2)
        
        # Plot probability contours
        contour = ax.contourf(xx, yy, Z, cmap='viridis', alpha=0.8, levels=np.linspace(0, 1, 11)) 
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='dashed')
        
        # Set labels with increased font size
        ax.set_xlabel(f'{f1_nice}', fontsize=font_size)
        ax.set_ylabel(f'{f2_nice}', fontsize=font_size)
        ax.set_title(f'Interaction: {f1_nice} × {f2_nice}', fontsize=font_size + 2)
        
        # Increase tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=font_size - 2)
        
        # Add colorbar to each subplot with increased font size
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Probability', fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size - 2)
    
    # Hide any unused subplots
    for idx in range(n_interactions, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    # Adjust spacing between subplots to accommodate larger font sizes
    plt.tight_layout(pad=3.0)
    
    # Save figure with a single filename
    filename = f"results/jailbreak_embedding_analysis/interaction_effects_combined"
    if suffix_model:
        filename += f"_{suffix_model}"
    filename += f"_{model_name}.png"
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    print(f"Combined interaction plot saved to {filename}")
    
    # Reset rcParams to default after saving to avoid affecting other plots
    plt.rcParams.update(plt.rcParamsDefault)
    
    return fig


if __name__ == "__main__":

    all_models = ["qwen2.5-3b-instruct", "vicuna-13b-v1.5", "llama-2-7b-chat-hf", "llama-3.2-1b-instruct"]

    suffix_model = None

    if suffix_model is None:
        for model_name in all_models: 
            file = f"{DATA_PATH}/{model_name}_layer_{MODEL_LAYER_MAPPING[model_name]}_suffix_push_refusal_connect_combined_multidim.json"
    
            with open(file, "r") as f:
                jailbreak_results = json.load(f)
    

    
            # Perform combined logistic regression analysis
            perform_logistic_regression(jailbreak_results, X_values=["refusal_connectivity", "suffix_push", "orthogonal_shift"], 
            interaction=["refusal_connectivity", "suffix_push", "refusal_connectivity", "orthogonal_shift", "suffix_push", "orthogonal_shift"],
            model_name = model_name,
            visualize_interaction=True)
        
            
            
            # Perform combined logistic regression analysis with semantic similarity
            perform_logistic_regression(jailbreak_results, X_values=["semantic_sim_model", "refusal_connectivity", "suffix_push", "orthogonal_shift"], 
            interaction=["refusal_connectivity", "suffix_push", 
            "refusal_connectivity", "orthogonal_shift", 
            "suffix_push", "orthogonal_shift", 
            "semantic_sim_model", "refusal_connectivity", 
            "semantic_sim_model","suffix_push",
            "semantic_sim_model", "orthogonal_shift"],
            model_name = model_name,
            visualize_interaction=False)

            # Perform single logistic regression analysis
            for variable in ["refusal_connectivity", "suffix_push", "orthogonal_shift"] 
            perform_logistic_regression(jailbreak_results, X_values=[variable], 
            interaction = [],
            model_name = model_name,
            visualize_interaction=False)


    suffixes_models = ["qwen2.5-3b-instruct", "llama-3.2-1b-instruct"]
    base_models = ["llama-3.2-1b-instruct", "qwen2.5-3b-instruct"]

    for suffix_model, base_model in zip(suffixes_models, base_models):

        file = f"{DATA_PATH}/{suffix_model}_{base_model}_layer_{MODEL_LAYER_MAPPING[base_model]}_suffix_push_refusal_connect_combined_multidim.json"

        with open(file, "r") as f:
            jailbreak_results = json.load(f)

        # Perform combined logistic regression analysis
        perform_logistic_regression(jailbreak_results, X_values=["refusal_connectivity", "suffix_push", "orthogonal_shift"], 
        interaction=["refusal_connectivity", "suffix_push", "refusal_connectivity", "orthogonal_shift", "suffix_push", "orthogonal_shift"],
        model_name = model_name,
        visualize_interaction=True)

    

    