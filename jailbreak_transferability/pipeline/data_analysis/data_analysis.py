import argparse
import os
import torch

import numpy as np
import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
import statsmodels.api as sm
import seaborn as sns

from scipy.stats import pearsonr  # Add this import for p-value calculation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from pipeline.config import Config
from pipeline.utils import utils
from pipeline.utils.activation_utils import get_prompt_activations, get_jailbreak_view

def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Parse arguments.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--dataset_parent_dir', type=str, required=False, help='The parent directory of the dataset. If not provided, it will be assumed that the directory is in the root folder.')
    return parser.parse_args()

def calculate_cosine_similarity(activations, refusal_direction):
    chunk_size = 10
    cosine_similarity_results = []

    for i in range(0, activations.shape[0], chunk_size):
        activations_chunk = activations[i:i+chunk_size]
        cosine_similarity = F.cosine_similarity(activations_chunk, refusal_direction.view(1, 1, 1, -1), dim=-1)
        cosine_similarity_results.append(cosine_similarity)
    
    return torch.cat(cosine_similarity_results, dim=0)

def print_most_and_least_successful_suffixes(success_matrix, n=5):
    most_successful_suffixes = success_matrix.sum(axis=0).nlargest(n)
    least_successful_suffixes = success_matrix.sum(axis=0).nsmallest(n)
    print(f"Most successful suffixes: {most_successful_suffixes}")
    print(f"Least successful suffixes: {least_successful_suffixes}")

def print_most_and_least_vulnerable_prompts(success_matrix, n=5):
    most_vulnerable_prompts = success_matrix.sum(axis=1).nlargest(n)
    least_vulnerable_prompts = success_matrix.sum(axis=1).nsmallest(n)
    print(f"Most vulnerable prompts: {most_vulnerable_prompts}")
    print(f"Least vulnerable prompts: {least_vulnerable_prompts}")

def plot_with_error_bars(layers, mean_values, std_values, color, label=None):
    """Helper function to plot mean values with error bars."""
    plt.plot(layers, mean_values, color=color, label=label)
    plt.fill_between(layers, mean_values - std_values, mean_values + std_values, color=color, alpha=0.2)

def compute_and_plot_linear_regression(x, y, xlabel, ylabel, title, save_path):
    """Helper function to compute linear regression, plot scatter and regression line."""
    correlation, p_value = pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)

    print(f"{title}:")
    print(f"Correlation: {correlation}, P-value: {p_value}")
    print(f"Linear Regression - Slope: {slope}, Intercept: {intercept}")

    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y)
    plt.plot(x, slope * x + intercept, color='red', label='Linear Regression')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_cosine_similarity_across_layers(activations, label, color, refusal_direction, num_layers):
    """Plot cosine similarity across layers with mean and standard deviation."""
    cosine_similarity = calculate_cosine_similarity(activations.unsqueeze(1), refusal_direction)[:, 0, :]
    mean_values = cosine_similarity.mean(dim=0).cpu().detach().numpy()
    std_values = cosine_similarity.std(dim=0).cpu().detach().numpy()
    layers = np.arange(num_layers)

    plot_with_error_bars(layers, mean_values, std_values, color, label)

def plot_cosine_similarity_for_prompts_and_suffix(prompt_activations, jailbreak_activations, refusal_direction, direction_num, num_layers, cfg, most_successful_suffix_ids, least_successful_suffix_ids):
    """Plot cosine similarity for prompts and multiple suffix activations."""
    plot_cosine_similarity_across_layers(
        activations=prompt_activations,
        label='Harmful Prompt',
        color='blue',
        refusal_direction=refusal_direction,
        num_layers=num_layers
    )

    for suffix_id in most_successful_suffix_ids:
        plot_cosine_similarity_across_layers(
            activations=jailbreak_activations[:, suffix_id, :, :],
            label=None,  # No individual label for each suffix
            color='green',
            refusal_direction=refusal_direction,
            num_layers=num_layers
        )

    for suffix_id in least_successful_suffix_ids:
        plot_cosine_similarity_across_layers(
            activations=jailbreak_activations[:, suffix_id, :, :],
            label=None,  # No individual label for each suffix
            color='orange',
            refusal_direction=refusal_direction,
            num_layers=num_layers
        )

    plt.xlabel("Layer number", fontsize=24)
    plt.ylabel("Cosine sim w refusal", fontsize=24)
    # plt.title(f"Average Cosine Similarity with Refusal Direction {direction_num} Across Layers")
    # plt.legend(handles=[
        # plt.Line2D([0], [0], color='blue', label='Harmful Prompt'),
        # plt.Line2D([0], [0], color='green', label='Harmful Prompt + Top 3 Most Successful Suffixes'),
        # plt.Line2D([0], [0], color='orange', label='Harmful Prompt + Top 3 Least Successful Suffixes')
    # ])
    plt.tight_layout()
    plt.savefig(f'figures/{cfg.model_alias}/cosine_similarity_across_layers_suffixes_direction_{direction_num}.png')
    plt.close()

def plot_cosine_similarity_vs_successful_jailbreaks(successful_jailbreaks, mean_cosine_similarity, dir_num, layer, cfg, type):
    """Plot and analyze the relationship between mean cosine similarity and successful jailbreaks."""
    xlabel = "Number of Prompts That Are Jailbroken by Suffix" if type == "suffix" else "Number of Suffixes that Jailbreak Prompt"
    ylabel = "Mean Cosine Similarity with Refusal Direction (Over Prompts)" if type == "suffix" else "Mean Cosine Similarity with Refusal Direction (Over Suffixes)"
    title = f"Mean Cosine Similarity of {type.capitalize()} With Refusal Direction {dir_num} vs Successful Jailbreaks per {type.capitalize()} (Layer {layer})"
    
    compute_and_plot_linear_regression(
        x=successful_jailbreaks,
        y=mean_cosine_similarity,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        save_path=f'figures/{cfg.model_alias}/cosine_similarity_vs_successful_jailbreaks_{type}_dir_{dir_num}.png'
    )

def plot_heatmap_with_borders(matrix, success_matrix, title, xlabel, ylabel, save_path, xticks, yticks, colorbar_label):
    """Helper function to plot a heatmap with red borders around successful jailbreaks."""
    plt.clf()
    plt.figure(figsize=(22, 22))
    plt.imshow(matrix.cpu().numpy(), cmap='viridis', aspect='equal')
    plt.colorbar(label=colorbar_label)
    # plt.title(title)
    plt.xticks(np.arange(len(xticks)), xticks, rotation=90)
    plt.yticks(np.arange(len(yticks)), yticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    ax = plt.gca()
    for i in range(success_matrix.shape[0]):  # rows
        for j in range(success_matrix.shape[1]):  # columns
            if success_matrix.iloc[i, j]:  # If the mask is True (or 1)
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

    plt.savefig(save_path)
    plt.close()

def plot_cosine_similarity_heatmap(similarity_matrix, success_matrix, prompts, suffixes, layer, dir_num, type, cfg):
    """Plot cosine similarity as a heatmap."""
    tick_labels = [str(i) for i in range(prompts)] + ["No Suffix"]
    plot_heatmap_with_borders(
        matrix=similarity_matrix,
        success_matrix=success_matrix,
        title=f"Cosine Similarity Of {type} Activations With Refusal Direction {dir_num} (Layer {layer})",
        xlabel="Suffix",
        ylabel="Prompt",
        save_path=f'figures/{cfg.model_alias}/cosine_similarity_with_refusal_heatmap_type_{type}_layer_{layer}_dir_{dir_num}.png',
        xticks=tick_labels,
        yticks=success_matrix.index.tolist(),
        colorbar_label="Cosine Similarity"
    )

def plot_suffix_directions_norm(suffix_ids, suffix_directions, num_layers, color, label):
    """Helper function to plot L2 norm of suffix directions."""
    for suffix_id in suffix_ids:
        suffix_directions_norm = torch.norm(suffix_directions, dim=-1)[:, suffix_id, :]
        suffix_directions_norm_mean = suffix_directions_norm.mean(dim=0).cpu().numpy()
        suffix_directions_norm_std = suffix_directions_norm.std(dim=0).cpu().numpy()
        layers = np.arange(num_layers)
        plot_with_error_bars(layers, suffix_directions_norm_mean, suffix_directions_norm_std, color, label if suffix_id == suffix_ids[0] else None)

def plot_suffix_directions_norm_across_layers(most_successful_suffix_ids, least_successful_suffix_ids, suffix_directions, num_layers, cfg):
    """Plot the L2 norm of suffix directions across layers for most and least successful suffixes."""
    plt.rcdefaults()
    plt.clf()
    plot_suffix_directions_norm(most_successful_suffix_ids, suffix_directions, num_layers, color='blue', label='Most successful suffixes')
    plot_suffix_directions_norm(least_successful_suffix_ids, suffix_directions, num_layers, color='orange', label='Least successful suffixes')

    plt.legend()
    plt.xlabel("Layer number")
    plt.ylabel("L2 Norm of Suffix Directions")
    # plt.title("Average L2 Norm of Suffix Directions Across Layers")
    plt.savefig(f'figures/{cfg.model_alias}/l2_norm_suffix_directions.png')
    plt.close()

def plot_suffix_directions_vs_successful_jailbreaks(suffix_directions, successful_jailbreaks_per_suffix, cfg):
    """Plot the relationship between L2 norm of suffix directions (last layer) and successful jailbreaks."""
    suffix_directions_norm_last_layer = torch.norm(suffix_directions, dim=-1)[:, :, -1]
    suffix_directions_norm_last_layer_mean = suffix_directions_norm_last_layer.mean(dim=0).cpu().numpy()

    compute_and_plot_linear_regression(
        x=successful_jailbreaks_per_suffix,
        y=suffix_directions_norm_last_layer_mean,
        xlabel="Number of Successful Jailbreaks",
        ylabel="L2 Norm of Suffix Directions (Last Layer)",
        title="L2 Norm of Suffix Directions vs Successful Jailbreaks",
        save_path=f'figures/{cfg.model_alias}/l2_norm_suffix_directions_vs_successful_jailbreaks.png'
    )

def plot_suffix_direction_norm_heatmap(matrix, success_matrix, cfg):
    """Plot L2 norm of suffix directions as a heatmap."""
    plot_heatmap_with_borders(
        matrix=matrix,
        success_matrix=success_matrix,
        title="L2 Norm of Suffix Directions (Last Layer)",
        xlabel="Suffix",
        ylabel="Prompt",
        save_path=f'figures/{cfg.model_alias}/l2_norm_suffix_directions_last_layer_heatmap.png',
        xticks=success_matrix.columns.tolist(),
        yticks=success_matrix.index.tolist(),
        colorbar_label="L2 Norm of Suffix Directions (Last Layer)"
    )

def compute_partial_correlation(df, x, y, covar):
    """
    Compute partial correlation between two variables controlling for a covariate.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x (str): Name of the first variable.
        y (str): Name of the second variable.
        covar (str): Name of the covariate.

    Returns:
        pd.DataFrame: Partial correlation result.
    """
    partial_corr = pg.partial_corr(data=df, x=x, y=y, covar=covar, method='pearson')
    print(f"Partial correlation between {x} and {y} controlling for {covar}:")
    print(partial_corr)
    return partial_corr

def fit_logistic_models_and_report(predictors, y):
    """Fit logistic regression models for multiple predictors and report results."""
    results = {}
    for model_name, X in predictors.items():
        log_model = LogisticRegression(solver='liblinear', class_weight='balanced')
        log_model.fit(X, y)
        preds = log_model.predict_proba(X)[:, 1]

        best_threshold = 0.5
        best_f1 = 0

        for threshold in np.arange(0.1, 1.0, 0.01):
            pred_labels = (preds >= threshold).astype(int)
            f1 = f1_score(y, pred_labels)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
        pred_labels = (preds >= best_threshold).astype(int)

        # Evaluate model
        auc = roc_auc_score(y, preds)
        report = classification_report(y, pred_labels)
        coef = log_model.coef_[0]
        odds_ratios = np.exp(coef)
        predictor_names = X.columns if isinstance(X, pd.DataFrame) else [f"Feature {i}" for i in range(X.shape[1])]
        odds_df = pd.DataFrame({
            'Predictor': predictor_names,
            'Coefficient': coef,
            'Odds Ratio': odds_ratios
        })

        results[model_name] = {
            "AUC": auc,
            "Classification Report": report,
            "Odds Ratios": odds_df
        }

    return results

def plot_scatter_with_regression(x, y, xlabel, ylabel, title, save_path):
    """Helper function to plot scatter with regression line and compute correlation."""
    correlation, p_value = pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)

    print(f"{title}:")
    print(f"Correlation: {correlation}, P-value: {p_value}")
    print(f"Linear Regression - Slope: {slope}, Intercept: {intercept}")

    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y)
    plt.plot(x, slope * x + intercept, color='red', label='Linear Regression')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def prepare_logistic_regression_data(success_matrix, middle_layer_similarities, suffix_directions_last_layer_norm, jailbreak_activations, layers, d_model):
    """Prepare data for logistic regression models."""
    log_df = pd.DataFrame({
        'success': success_matrix.to_numpy().flatten(),
        'norm': suffix_directions_last_layer_norm.cpu().flatten()
    })

    # Dynamically add cosine similarity columns for all directions
    for i, similarity in enumerate(middle_layer_similarities):
        log_df[f'cosine_similarity_dir_{i}'] = similarity.cpu().flatten()

    predictors = {
        f"Model {i + 1} (Cosine Similarity with Direction {i})": log_df[[f'cosine_similarity_dir_{i}']]
        for i in range(len(middle_layer_similarities))
    }

    predictors.update({
        f"Model {len(middle_layer_similarities) + 1} (Norm)": log_df[['norm']],
        f"Model {len(middle_layer_similarities) + 2} (Cosine Similarity with All Directions + Norm)": log_df[
            [f'cosine_similarity_dir_{i}' for i in range(len(middle_layer_similarities))] + ['norm']
        ],
        f"Model {len(middle_layer_similarities) + 3} (Activations at Layer {layers[0]})": jailbreak_activations[:, :, layers[0], :].reshape(-1, d_model).cpu().numpy(),
    })

    # Add models combining activations with each direction
    for i in range(len(middle_layer_similarities)):
        predictors[f"Model {len(predictors) + 1} (Activations at Layer {layers[0]} + Refusal Direction {i})"] = np.hstack([
            jailbreak_activations[:, :, layers[0], :].reshape(-1, d_model).cpu().numpy(),
            log_df[[f'cosine_similarity_dir_{i}']]
        ])

    return log_df['success'], predictors

def evaluate_and_print_logistic_models(predictors, y):
    """Fit logistic regression models and print results."""
    # TODO: refactor out test_size and random_state to pass them as inputs
    test_size = 0.2
    random_state = 42

    results = {}

    for model_name, X in predictors.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = LogisticRegression(random_state=random_state, max_iter=10000, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        class_report = classification_report(y_test, y_pred, zero_division=0)

        odds_ratios = np.exp(model.coef_) if X.ndim <= 2 and X.shape[1] <= 3 else None

        results[model_name] = {
            'AUC': auc,
            'Classification Report': class_report,
            'Odds Ratios': odds_ratios
        }


    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"AUC: {result['AUC']}")
        print("Classification Report:\n", result['Classification Report'])
        if result['Odds Ratios'] is not None:
            print("Odds Ratios:\n", result['Odds Ratios'])

def plot_and_save(title, xlabel, ylabel, save_path, legend_handles=None):
    """Helper function to finalize and save plots."""
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title(title)
    if legend_handles:
        plt.legend(handles=legend_handles)
    plt.savefig(save_path)
    plt.close()

def compute_and_plot_relationship(x, y, xlabel, ylabel, title, save_path):
    """Helper function to compute correlation and plot scatter with regression line."""
    correlation, p_value = pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)

    print(f"{title}:")
    print(f"Correlation: {correlation}, P-value: {p_value}")
    print(f"Linear Regression - Slope: {slope}, Intercept: {intercept}")

    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y)
    plt.plot(x, slope * x + intercept, color='red', label='Linear Regression')
    plot_and_save(title, xlabel, ylabel, save_path, legend_handles=[plt.Line2D([0], [0], color='red', label='Linear Regression')])

def evaluate_linear_models(df, feature_sets):
    """
    Evaluate linear regression models for given feature sets and print results.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        feature_sets (list of dict): List of feature set dictionaries. Each dictionary should have:
            - 'features': List of feature column names.
            - 'label': Label for the feature set (used in print statements).
    """
    y = df['jailbreak_success']

    for feature_set in feature_sets:
        X = df[feature_set['features']]
        X = sm.add_constant(X)  # Add constant term for intercept
        model = sm.OLS(y, X).fit()
        print(feature_set)
        print(model.summary())

def evaluate_linear_models_cos_sim(df, num_directions):
    """
    Evaluate linear regression models for cosine similarity with multiple directions.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        num_directions (int): Number of directions to include in the models.
    """
    feature_sets = []

    # Add individual direction models
    for i in range(num_directions):
        feature_sets.append({
            'features': [f'cosine_similarity_dir_{i}'],
            'label': f'cosine similarity with direction {i}'
        })

    # Add combined model with all directions
    if num_directions > 1:
        all_directions = [f'cosine_similarity_dir_{i}' for i in range(num_directions)]
        feature_sets.append({
            'features': all_directions,
            'label': 'cosine similarity with all directions'
        })

    evaluate_linear_models(df, feature_sets)

def evaluate_logistic_models(success_matrix, middle_layer_similarities, suffix_directions_last_layer_norm, jailbreak_activations, layers, d_model):
    """Prepare and evaluate logistic regression models."""
    y, predictors = prepare_logistic_regression_data(
        success_matrix=success_matrix,
        middle_layer_similarities=middle_layer_similarities,
        suffix_directions_last_layer_norm=suffix_directions_last_layer_norm,
        jailbreak_activations=jailbreak_activations,
        layers=layers,
        d_model=d_model
    )
    evaluate_and_print_logistic_models(predictors, y)

def data_analysis(cfg):
    seed = 42
    np.random.seed(seed)

    transfer_df = utils.get_transfer_df(cfg)
    success_matrix = utils.get_jailbreak_success_matrix_df(transfer_df)
    prompt_activations = get_prompt_activations(cfg).to('cuda')
    jailbreak_activations = get_jailbreak_view(cfg).to('cuda')
    num_prompts, num_suffixes, num_layers, d_model = jailbreak_activations.shape
    print("Shape of activations:", prompt_activations.shape, jailbreak_activations.shape)

    print(len(transfer_df))
    print(success_matrix.shape)
    print(prompt_activations.shape)
    print(jailbreak_activations.shape)

    refusal_directions = utils.get_refusal_directions(cfg.arditi_et_al_refusal_direction_dir())

    if len(refusal_directions) > 1:
        refusal_directions_stacked = torch.stack(refusal_directions)
        cosine_sim_matrix = F.cosine_similarity(refusal_directions_stacked.unsqueeze(1), refusal_directions_stacked.unsqueeze(0), dim=-1)
        print("Pairwise Cosine Similarity Matrix Between Refusal Directions:")
        print(cosine_sim_matrix)

    # Print most and least successful suffixes and vulnerable prompts + plot success matrix
    print_most_and_least_successful_suffixes(success_matrix)
    print_most_and_least_vulnerable_prompts(success_matrix)
    utils.plot_jailbreak_success_matrix(success_matrix, cfg, x_label="Suffix", y_label="Prompt")

    # Get the most and least successful suffix IDs
    most_successful_suffix_ids = success_matrix.sum(axis=0).nlargest(3).index.tolist()  # Top 3 suffixes
    least_successful_suffix_ids = success_matrix.sum(axis=0).nsmallest(3).index.tolist()  # Bottom 3 suffixes
    print(f"Most successful suffix IDs: {most_successful_suffix_ids}")
    print(f"Least successful suffix IDs: {least_successful_suffix_ids}")

    # Plot the cosine similarity with refusal direction for the most and least successful suffixes
    for i, refusal_direction in enumerate(refusal_directions):
        plot_cosine_similarity_for_prompts_and_suffix(
            prompt_activations=prompt_activations,
            jailbreak_activations=jailbreak_activations,
            refusal_direction=refusal_direction, 
            direction_num=i,
            num_layers=num_layers,
            cfg=cfg,
            most_successful_suffix_ids=most_successful_suffix_ids,
            least_successful_suffix_ids=least_successful_suffix_ids
        )

    # Calculate cosine similarity for jailbreak activations and extract middle layer

    # TODO: loop over the layers and pick the one with the highest correlation
    # layers = [14, 22] # The layers the refusal direction is strongest in
    layers = [cfg.arditi_et_al_refusal_direction_layer(dir_num=0)]

    jailbreak_cosine_similarities = [calculate_cosine_similarity(jailbreak_activations, d) for d in refusal_directions]
    middle_layer_similarities = [jailbreak_cosine_similarities[i][:, :, layers[i]] for i in range(len(refusal_directions))]
    mean_cosine_similarities_over_prompts = [m.mean(dim=0).cpu().numpy() for m in middle_layer_similarities]

    # Calculate the number of successful jailbreaks per suffix
    successful_jailbreaks_per_suffix = success_matrix.sum(axis=0)

    # Plot the relationship between mean cosine similarity of a suffix and the number of prompts it jailbreaks
    for i in range(len(refusal_directions)):
        plot_cosine_similarity_vs_successful_jailbreaks(
            successful_jailbreaks=successful_jailbreaks_per_suffix,
            mean_cosine_similarity=mean_cosine_similarities_over_prompts[i],
            layer=layers[i],
            dir_num=i,
            type="suffix",
            cfg=cfg
        )

    # Compute the partial correlation between the outcome (successful jailbreaks per suffix and the cosine similarity 
    # to the second refusal direction while controlling for the cosine similarity with the first direction.
    data = {
        'jailbreak_success': successful_jailbreaks_per_suffix
    }
    for i in range(len(refusal_directions)):
        data[f'cosine_similarity_dir_{i}'] = mean_cosine_similarities_over_prompts[i]

    df = pd.DataFrame(data)

    for i in range(1, len(refusal_directions)):
        compute_partial_correlation(
            df, 
            x=f'cosine_similarity_dir_{i}', 
            y='jailbreak_success', 
            covar=f'cosine_similarity_dir_{i-1}'
        )

    evaluate_linear_models_cos_sim(df, num_directions=len(refusal_directions))

    suffix_directions = jailbreak_activations - prompt_activations.unsqueeze(1)
    suffix_directions_last_layer_norm = torch.norm(suffix_directions, dim=-1)[:, :, -1]

    # Evaluate logistic regression models
    evaluate_logistic_models(
        success_matrix=success_matrix,
        middle_layer_similarities=middle_layer_similarities,
        suffix_directions_last_layer_norm=suffix_directions_last_layer_norm,
        jailbreak_activations=jailbreak_activations,
        layers=layers,
        d_model=d_model
    )

    for i in range(len(refusal_directions)):
        prompt_similarity = calculate_cosine_similarity(prompt_activations.unsqueeze(1), refusal_directions[0])[:, :, layers[i]]
        middle_layer_similarity_plus_no_suffix = torch.cat([middle_layer_similarities[i], prompt_similarity], dim=1)

        plot_cosine_similarity_heatmap(
            similarity_matrix=middle_layer_similarity_plus_no_suffix,
            success_matrix=success_matrix,
            prompts=num_prompts,
            suffixes=num_suffixes,
            layer=layers[i],
            dir_num=i,
            type="Prompt + Suffix",
            cfg=cfg
        )
    
    # Add a helper function to utils to get the prompts_df but only the ones that were previously refused
    previously_refused_indices = utils.get_previously_refused_indices(cfg)
    prompts_df = pd.read_json(cfg.prompts_path())
    prompt_to_cat = prompts_df.set_index('prompt_id')['category'].loc[previously_refused_indices]
    suffix_to_cat = prompt_to_cat.reindex(success_matrix.columns)

    P = pd.get_dummies(prompt_to_cat).sort_index()
    S = pd.get_dummies(suffix_to_cat).sort_index()

    succ = P.T.dot(success_matrix).dot(S)

    nP = P.sum(axis=0)
    nS = S.sum(axis=0)

    rate = succ.div(nP, axis=0).div(nS, axis=1)
    rate = rate.fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(rate, annot=True, fmt=".2f", cmap='YlGnBu_r', cbar=True, xticklabels=False, yticklabels=False, ax=ax)
    ax.set_xticks(np.arange(rate.shape[1]) + 0.5)
    ax.set_yticks(np.arange(rate.shape[0]) + 0.5)
    ax.set_xticklabels(rate.columns, rotation=90)
    ax.set_yticklabels(rate.index, rotation=0)
    ax.set_xlabel('Suffix Category')
    ax.set_ylabel('Prompt Category')
    # ax.set_title(f'{cfg.model_name().capitalize()} Normalized Per-Group Jailbreak Success Rate')
    fig.tight_layout()
    plt.savefig(cfg.per_group_success_matrix_figure_path())
    plt.close()



if __name__ == "__main__":
    args = parse_arguments()
    cfg = Config(model_path=args.model_path, dataset_parent_dir=args.dataset_parent_dir)
    data_analysis(cfg)