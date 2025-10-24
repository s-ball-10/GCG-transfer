
import os
import json
import pandas as pd

class Config:
    def __init__(self, model_path: str, dataset_parent_dir: str = None):
        self.dataset_parent_dir = dataset_parent_dir if dataset_parent_dir else ''
        self.model_path = model_path
        self.model_alias = self._get_model_alias(model_path)
        self.DATASET_DIR = "jailbreak_transferability_data/dataset"
        self.REFUSAL_DIR_PATH = "jailbreak_transferability_data/refusal_directions"
        self.PROCESSED_DATASET_DIR = os.path.join(self.DATASET_DIR, "processed")
        self.FIGURES = "figures"
        self.num_suffixes_per_prompt = 100
        self.num_prompts = len(pd.read_json(self.prompts_path()))

    def _model_name(self, model_alias) -> str:
        if "llama-2" in model_alias:
            return "Llama 2"
        elif "llama-3.2" in model_alias:
            return "Llama 3.2"
        elif "vicuna" in model_alias:
            return "Vicuna"
        elif "qwen2.5" in model_alias:
            return "Qwen 2.5"
    
    def model_name(self) -> str:
        return self._model_name(self.model_alias)

    def prompts_path(self) -> str:
        return os.path.join(self.dataset_parent_dir, self.PROCESSED_DATASET_DIR, "jailbreakbench_prompts", "prompts.json")
    
    def suffixes_path(self) -> str:
        return os.path.join(self.dataset_parent_dir, self.PROCESSED_DATASET_DIR, "jailbreakbench_suffixes", f"{self.model_alias}_suffixes.json")
    
    def cross_prompt_transfer_generations_path(self) -> str:
        return os.path.join(self.dataset_parent_dir, self.DATASET_DIR, "cross_prompt_transfer_generations", f"{self.model_alias}_cross_prompt_transfer_generations.json")
    
    def cross_model_transfer_generations_dir(self) -> str:
        return os.path.join(self.dataset_parent_dir, self.DATASET_DIR, "cross_model_transfer_generations")
    
    def no_suffix_generations_path(self) -> str:
        return os.path.join(self.dataset_parent_dir, self.DATASET_DIR, "no_suffix_generations", f"{self.model_alias}_no_suffix_generations.json")
    
    def multi_seed_generations_no_transfer_path(self) -> str:
        return os.path.join(self.dataset_parent_dir, self.DATASET_DIR, "multiple_seed_results", self.model_alias, "no_transfer", f"{self.model_alias}_multiple_seed_results_no_transfer.json")

    def multi_seed_generations_transfer_path(self) -> str:
        return os.path.join(self.dataset_parent_dir, self.DATASET_DIR, "multiple_seed_results", self.model_alias, "transfer", f"{self.model_alias}_multiple_seed_results_transfer.json")
    
    def multi_seed_generations_transfer_dir(self) -> str:
        return os.path.join(self.dataset_parent_dir, self.DATASET_DIR, "multiple_seed_results", self.model_alias, "transfer")

    def activations_dir(self) -> str:
        return os.path.join(self.dataset_parent_dir, self.DATASET_DIR, "activations")

    def prompt_activations_path(self) -> str:
        return os.path.join(self.activations_dir(), self.model_alias, "prompt_activations", "prompt_activations.pt")
    
    def suffix_activations_path(self) -> str:
        return os.path.join(self.activations_dir(), self.model_alias, "suffix_activations", "suffix_activations.pt")
    
    def jailbreak_activations_dir(self) -> str:
        return os.path.join(self.activations_dir(), self.model_alias, "jailbreak_activations")
    
    def multi_seed_jailbreak_activations_transfer_dir(self) -> str:
        return os.path.join(self.activations_dir(), self.model_alias, "multi_seed_jailbreak_activations_transfer")
    
    def s_activations_dir(self) -> str:
        return os.path.join(self.activations_dir(), self.model_alias, "s_activations")
    
    def s_jailbreak_activations_dir(self) -> str:
        return os.path.join(self.s_activations_dir(), "jailbreak_activations")
    
    def s_prompt_activations_dir(self) -> str:
        return os.path.join(self.s_activations_dir(), "prompt_activations")
    
    def s_jailbreak_activations_path(self) -> str:
        return os.path.join(self.s_jailbreak_activations_dir(), f'{self.model_alias}_s_jailbreak_activations.pt')
    
    def s_prompt_activations_path(self) -> str:
        return os.path.join(self.s_prompt_activations_dir(), f'{self.model_alias}_s_prompt_activations.pt')

    def cosine_similarity_with_refusal_path(self) -> str:
        return os.path.join(self.dataset_parent_dir, self.DATASET_DIR, "activations", self.model_alias, "cosine_sim_with_refusal", "cosine_similarity_with_refusal.pt")

    def jailbreak_success_matrix_figure_path(self) -> str:
        return os.path.join(self.FIGURES, self.model_alias, "jailbreak_success_matrix.png")
    
    def multi_seed_jailbreak_success_matrix_figure_path(self) -> str:
        return os.path.join(self.FIGURES, self.model_alias, "multi_seed_jailbreak_success_matrix.png")
    
    def per_group_success_matrix_figure_path(self) -> str:
        return os.path.join(self.FIGURES, self.model_alias, "per_group_success_matrix.png")
    
    def arditi_et_al_refusal_direction_dir(self) -> str:
        return os.path.join(self.dataset_parent_dir, self.REFUSAL_DIR_PATH, "arditi_et_al_2024", self.model_alias)
    
    def arditi_et_al_refusal_direction_layer(self, dir_num) -> str:
        dir = self.arditi_et_al_refusal_direction_dir()
        file = f'direction_{dir_num}_metadata.json'
        with open(os.path.join(dir, file), 'r') as f:
            data = json.load(f)
        return data['layer']
    
    def refusal_direction_dir(self) -> str:
        return os.path.join(self.dataset_parent_dir, self.REFUSAL_DIR_PATH, "ours", self.model_alias)

    def _get_model_alias(self, model_path: str):
        return os.path.basename(model_path).lower()