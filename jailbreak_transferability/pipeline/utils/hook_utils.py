import torch
import contextlib
import functools

from typing import List, Tuple, Callable


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """ 
    This is a context manager that temporarily adds forward hooks to a model.
    The context manager will add the hooks to the model and then remove them after the block of code is executed.
    The context manager guarantees that the hooks are only active during the specific block of code where they're needed.
    Allows for running the code as follows:
    `with add_hooks(...):`
        `# code to run with hooks`

    Parameters
    ----------
    module_forward_pre_hooks : List[Tuple[torch.nn.Module, Callable]]
        A list of pairs: (module, fnc) The function will be registered as a forward pre hook on the module
        Run the function before the module's forward pass
        Receives the module and its input, allowing for modification of the input before it's passed to the module
    module_forward_hooks : List[Tuple[torch.nn.Module, Callable]]
        A list of pairs: (module, fnc) The function will be registered as a forward hook on the module
        Run the function after the module's forward pass
        Receives the module, its input, and its output, allowing for inspecting the output of the module
    """

    try:
        # Code to run before the block of code inside the `with` statement
        handles = []
        for module, hook in module_forward_pre_hooks:
            # Create a partial function that calls the hook function with the specified keyword arguments
            # functools.partial creates a new function with the same code as the original function, but with some arguments fixed
            partial_hook = functools.partial(hook, **kwargs)
            # module.register_forward_pre_hook expects a function that accepts two arguments: 
            # 1) the module and 2) the inputs that will be passed to the module's forward method
            # If the hook returns a non-None value, the return value will replace the original input to the module's forward method
            # module.register_forward_pre_hook returns a handle that can be used to remove the hook later
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            # module.register_forward_hook expects a function that accepts three arguments:
            # 1) the module, 
            # 2) the inputs that will be passed to the module's forward method, 
            # 3) the output of the module's forward method
            handles.append(module.register_forward_hook(partial_hook))
        yield # This is where the code block inside the `with` statement will run
    finally: # This block of code will always run, even if an exception is raised
        # Code to run after the block of code inside the `with` statement
        for h in handles:
            h.remove()

