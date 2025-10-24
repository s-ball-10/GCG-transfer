import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from typing import List, Callable, Any

from pipeline.model_utils.model_base import ModelBase

def steering_hook_factory(
    steering_vec: Float[Tensor, "d_model"], coeff: float, pre: bool = False
) -> Callable:
    """
    Returns a hook function for modifying activations.
    If `pre` is True, the hook is applied to the inputs (pre-activation);
    otherwise, it is applied to the outputs (post-activation).
    """
    def hook_fn(module, *args):
        # For pre-hook, args[0] is the input tuple;
        # For post-hook, args[1] is the output tuple.
        activation: Float[Tensor, "batch_size seq_len d_model"] = args[0][0] if pre else args[1][0]
        steering = steering_vec.to(activation.device, dtype=activation.dtype)
        activation.add_(coeff * steering)
    return hook_fn

def steer_activations(
    model: ModelBase,
    steering_vec: Float[Tensor, "d_model"],
    texts: List[str],
    block_modules: List[torch.nn.Module],
    layer: int = -1,
    coeff: float = 1.0,
    batch_size: int = 32,
    pre: bool = False,
    max_new_tokens: int = 64
) -> List[Any]:
    """
    Apply a steering hook to the specified layer's activations (pre or post)
    and generate completions for the provided texts.
    
    Parameters:
        model: The model to steer.
        steering_vec: The vector used to steer activations.
        texts: A list of input texts.
        block_modules: List of modules (e.g., transformer blocks) in the model.
        layer: The index of the layer to attach the hook (default: -1, the last layer).
        coeff: Scaling coefficient for the steering vector.
        batch_size: Batch size for generating completions.
        hook_type: 'post' for a post-activation hook (default) or 'pre' for a pre-activation hook.
    
    Returns:
        A list of generated completions.
    """
    torch.cuda.empty_cache()
    hook_type = 'pre' if pre else 'post'
    
    # Mapping to determine pre-hook vs. post-hook settings.
    hook_configs = {
        'post': (False, "fwd_hooks"),
        'pre': (True, "fwd_pre_hooks")
    }
    
    pre_flag, hook_arg = hook_configs[hook_type]
    hook = steering_hook_factory(steering_vec, coeff, pre=pre_flag)
    hooks = [(block_modules[layer], hook)]
    
    completions = []
    for i in tqdm(range(0, len(texts), batch_size)):
        texts_batch = texts[i:i+batch_size]
        completions_batch = model.generate_completions(
            texts_batch, **{hook_arg: hooks, "batch_size": batch_size, "max_new_tokens": max_new_tokens}
        )
        completions.extend(completions_batch)
    return completions