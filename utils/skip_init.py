from contextlib import contextmanager

import torch
import transformers


@contextmanager
def skip_init(dtype=torch.float16):
    """Context manager to temporarily disable init for models.
    This is a nasty hack, but it speeds up model building by a huge amount.
    """
    old_default_dtype = torch.get_default_dtype()
    transformers.modeling_utils._init_weights = False

    torch.set_default_dtype(dtype)

    def skip(*args, **kwargs):
        pass

    old_inits = (
        torch.nn.init.kaiming_uniform_,
        torch.nn.init.uniform_,
        torch.nn.init.normal_,
    )
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    yield

    # Restore the old initializers
    (
        torch.nn.init.kaiming_uniform_,
        torch.nn.init.uniform_,
        torch.nn.init.normal_,
    ) = old_inits

    torch.set_default_dtype(old_default_dtype)
    transformers.modeling_utils._init_weights = True
