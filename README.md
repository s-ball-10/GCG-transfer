# Toward Understanding the Transferability of Adversarial Suffixes in Large Language Models 

This repository contains code for the paper *Toward Understanding the Transferability of Adversarial Suffixes in Large Language Models*

## Prerequisites

**Note:** The main experiments for the paper (in folder `data_analysis`) can work on a single CPU. Generating the GCG suffixes, the model responses and the jailbreak judge evaluations is compute-intensive and may require tinkering with to get set up on your compute server of choice. Most scripts contain options to parallelize the process on multiple GPUs (by setting `--chunk_id`).

**GPU Required:** Generation of GCG suffixes and model evaluations require GPU resources.

## Supported Models

We support the following models and access them via HuggingFace:
- `meta-llama/Llama-2-7b-chat-hf`
- `lmsys/vicuna-13b-v1.5`
- `meta-llama/Llama-3.2-1B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`

## Setup

### Environment Installation

1. Set up conda environment for GCG generation:
   ```bash
   cd generating_GCG_suffixes
   conda create -n jbb
   conda activate jbb
   pip install jailbreakbench nanogcg
   ```

2. Set up conda environment for jailbreak transferability:
   ```bash
   cd ../jailbreak_transferability
   conda env create -f environment.yml
   conda activate jailbreak_transferability_env
   ```

### Configuration

3. Export model path:
   ```bash
   cd ..
   export MODEL_ID=model/id
   ```

## Running the Pipeline

This section details how to run the full pipeline for generating GCG suffixes, evaluating suffix transferability, and analyzing the results.

### 1. Generate GCG Suffixes

Create GCG suffixes for JailbreakBench prompts:

```bash
# Run GCG generation (example given in run_locally.sh)
python generating_GCG_suffixes/gcg.py

# Create dataset
python generating_GCG_suffixes/create_dataset.py --model_path $MODEL_ID
```

### 2. Organize GCG Suffixes

Move generated suffixes to the appropriate directories:

```bash
mkdir -p jailbreak_transferability/jailbreak_transferability_data/dataset/multiple_seed_results/$MODEL_ID/transfer
mkdir -p jailbreak_transferability/jailbreak_transferability_data/dataset/multiple_seed_results/$MODEL_ID/no_transfer

mv ${MODEL_ID}_multiple_seed_results_transfer.json jailbreak_transferability/jailbreak_transferability_data/dataset/multiple_seed_results/$MODEL_ID/transfer
mv ${MODEL_ID}_multiple_seed_results_no_transfer.json jailbreak_transferability/jailbreak_transferability_data/dataset/multiple_seed_results/$MODEL_ID/no_transfer
```

### 3. Generate Model Completions

Generate completions for the prompts with and without suffixes:

```bash
cd jailbreak_transferability
python3 -m pipeline.generate_completions --model_path $MODEL_ID --multi_seed --no_suffix_completions
```

### 4. Evaluate with Jailbreak Judge

Evaluate the jailbreak judge for the generated completions:

```bash
# Evaluate with suffixes
python3 -m pipeline.evaluate_completions --model_path $MODEL_ID --num_gpus 4 --multi_seed

# Evaluate without suffixes
python3 -m pipeline.evaluate_completions --model_path $MODEL_ID --num_gpus 4 --no_suffix_completions
```

### 5. Extract Activations

Extract model activations for analysis:

```bash
# Extract activations with multi-seed prompts
python3 -m pipeline.save_activations --model_path $MODEL_ID --multi_seed --prompts --jailbreak

# Extract activations with s parameter
python3 -m pipeline.save_activations --model_path $MODEL_ID --s --prompts --jailbreak --multi_seed
```

## Data Analysis

This section explains how to analyze the jailbreak transferability results.

### Refusal Direction Analysis

Extract refusal directions from the following codebase: https://github.com/andyrdt/refusal_direction

### Analysis Scripts

Run the following analysis scripts:

```bash
# Multi-seed data analysis
python3 -m pipeline.data_analysis.multi_seed_data_analysis --model_path $MODEL_ID

# Dispersion plot
python3 -m pipeline.data_analysis.dispersion_plot

# Semantic similarity analysis
python3 -m pipeline.data_analysis.semantic_similarity

# Suffix push analysis
python3 -m pipeline.data_analysis.suffix_push

# Logistic regression analysis
python3 -m pipeline.data_analysis.logistic_regression
```

## Cross-Model Analysis

This section details how to analyze jailbreak transferability across different models.

### Setup

Configure source and target models:

```bash
export SOURCE_MODEL_ID=source/model/id
export TARGET_MODEL_ID=target/model/id
```

### Running Cross-Model Pipeline

```bash
# Save one suffix per prompt from source model
python3 -m pipeline.cross_model.save_one_suffix_per_prompt --model_path $SOURCE_MODEL_ID

# Set up cross-model dataset
python3 -m pipeline.cross_model.set_up_dataset --source_model_path $SOURCE_MODEL_ID --target_model_path $TARGET_MODEL_ID

# Generate completions on target model
python3 -m pipeline.cross_model.generate_completions --source_model_path $SOURCE_MODEL_ID --target_model_path $TARGET_MODEL_ID

# Evaluate completions
python3 -m pipeline.cross_model.evaluate_completions --source_model_path $SOURCE_MODEL_ID --target_model_path $TARGET_MODEL_ID

# Extract activations
python3 -m pipeline.cross_model.save_activations --source_model_path $SOURCE_MODEL_ID --target_model_path $TARGET_MODEL_ID

# Extract activations with s parameter
python3 -m pipeline.cross_model.save_activations --source_model_path $SOURCE_MODEL_ID --target_model_path $TARGET_MODEL_ID --s

# Run cross-model data analysis
python3 -m pipeline.cross_model.data_analysis --source_model_path $SOURCE_MODEL_ID --target_model_path $TARGET_MODEL_ID
```

## Citation

If you like our work and find it useful, please cite:

```bibtex
@article{ball2025toward,
  title = {Toward Understanding the Transferability of Adversarial Suffixes in Large Language Models },
  author = {Ball, S., Hasrati, N., Robey, A., Schwarzschild, A., Kreuter, F., Kolter, Z. & Risteski, A.},
  journal = {arXiv preprint arXiv:xxxx.xxxxx},
  year = {2025}
}
```