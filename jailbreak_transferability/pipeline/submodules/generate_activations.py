import torch 

from jaxtyping import Float
from torch import Tensor
from typing import List, Callable, Dict
from tqdm import tqdm

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase

# Code adapter from https://github.com/andyrdt/refusal_direction/blob/main/pipeline/submodules/generate_directions.py

def compute_last_indices_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    # Sum over tokens to get the real length, then subtract 1 for the last index.
    return attention_mask.sum(dim=1) - 1

def activations_hook_factory(
    layer: int,
    cache: Float[Tensor, "batch_size layer d_model"],
    hook_state: Dict[str, torch.Tensor],
    aggregation: str = "last",  # "last" for last token or "mean" for mean over tokens
    pre: bool = False,
    batch_offset: int = 0,
) -> Callable:
    """
    Returns a hook function that captures activations for a given layer.
    
    Parameters:
        layer: The layer index.
        cache: A tensor to store the activations.
        aggregation: 'last' to capture the last token activation,
                     or 'mean' to capture the mean over tokens.
        pre: If True, apply the hook to inputs (pre-activation); otherwise, outputs (post-activation).
    """
    def hook_fn(module, *args):
        # For pre-hooks, args[0] is the input tuple.
        # For post-hooks, args[1] is the output tuple.
        # Activation has shape (batch_size, seq_len, d_model)
        activation = args[0][0].clone().to(cache) if pre else args[1][0].clone().to(cache)
        if aggregation == "last":
            last_indices = hook_state["last_indices"]
            batch_size, seq_len, d_model = activation.shape
            last_index_expanded = last_indices.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, d_model)
            value = activation.gather(dim=1, index=last_index_expanded).squeeze(1)
            # value = activation[:, -1, :]
        elif aggregation == "mean":
            # TODO: Only take mean over "real" tokens (exclude padding)
            value = activation.mean(dim=1)
        else:
            raise ValueError("aggregation must be either 'last' or 'mean'")
        batch_size = value.size(0)
        cache[batch_offset: batch_offset + batch_size, layer, :] = value
    return hook_fn

def get_activations(
    model: ModelBase,
    texts: List[str],
    block_modules: List[torch.nn.Module],
    aggregation: str = "last",
    pre: bool = False,
    batch_size: int = 100,
) -> Float[Tensor, "len(texts) num_layers d_model"]:
    """
    Capture activations from all layers using hooks.

    Parameters:
        model: The model from which activations are captured.
        texts: A list of input texts.
        block_modules: List of modules (e.g., transformer blocks) to attach hooks.
        aggregation: 'last' to capture the last token's activations or 'mean' for the mean over tokens.
        pre: If True, capture activations pre-forward; otherwise, post-forward.
        batch_size: Batch size for processing texts.

    Returns:
        A tensor of shape (len(texts), num_layers, d_model) containing captured activations.
    """
    torch.cuda.empty_cache()

    model.tokenizer.padding_side = "right"

    n_layers = model.model.config.num_hidden_layers
    d_model = model.model.config.hidden_size

    # Allocate a high-precision cache for numerical stability.
    activations_cache = torch.zeros(
        (len(texts), n_layers, d_model), dtype=torch.float64, device=model.model.device
    )

    hook_state: Dict[str, torch.Tensor] = {"last_indices": None}

    for i in range(0, len(texts), batch_size):
        texts_batch = texts[i:i+batch_size]
        tokenized_texts = model.tokenize_instructions_fn(instructions=texts_batch)
        attention_mask = tokenized_texts.attention_mask.to(model.model.device)
        hook_state["last_indices"] = compute_last_indices_from_mask(attention_mask)

        # Create a hook for each layer.
        hooks = [
            (block_modules[layer], activations_hook_factory(layer, activations_cache, hook_state, aggregation, pre, batch_offset=i))
            for layer in range(n_layers)
        ]

        # Select the correct hook keyword arguments.
        hook_kwargs = (
            {"module_forward_pre_hooks": hooks, "module_forward_hooks": []} if pre 
            else {"module_forward_pre_hooks": [], "module_forward_hooks": hooks}
        )

        with torch.no_grad():
            with add_hooks(**hook_kwargs):
                model.model(
                    input_ids=tokenized_texts.input_ids.to(model.model.device),
                    attention_mask=attention_mask,
                )

        # torch.cuda.empty_cache()

    return activations_cache