from __future__ import annotations

import warnings


def suppress_grounded_sam2_warnings() -> None:
    warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint")
    warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
    warnings.filterwarnings("ignore", message="torch.meshgrid")
    warnings.filterwarnings("ignore", message="Memory efficient kernel not used")
    warnings.filterwarnings("ignore", message="Memory Efficient attention")
    warnings.filterwarnings("ignore", message="Flash attention kernel not used")
    warnings.filterwarnings("ignore", message="Expected query, key and value to all be of dtype")
    warnings.filterwarnings("ignore", message="cuDNN attention kernel not used")
    warnings.filterwarnings("ignore", message="Flash Attention kernel failed")
    warnings.filterwarnings("ignore", message="Falling back to all available kernels")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\.models\.layers")

