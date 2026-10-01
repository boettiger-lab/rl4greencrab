"""
Detect which numeric/performance modes the current device supports, and fall back gracefully.

TF32 tensor-core matmuls need compute capability >= 8.0 (Ampere or newer); on older GPUs
such as Turing (e.g. Quadro RTX 8000, sm_75) or on CPU, requests for TF32 fall back to
full fp32. CUDA graphs need a CUDA device.
"""

import warnings

import torch


def device_capabilities(device=None):
    """Dict describing the device: name, compute capability, and support for TF32, BF16 and CUDA graphs."""
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" or not torch.cuda.is_available():
        return {"device": str(dev), "compute_capability": None, "tf32": False, "bf16": False, "cuda_graphs": False}
    cc = torch.cuda.get_device_capability(dev)
    return {
        "device": torch.cuda.get_device_name(dev),
        "compute_capability": cc,
        "tf32": cc >= (8, 0),
        "bf16": torch.cuda.is_bf16_supported(),
        "cuda_graphs": True,
    }


def resolve(request, supported, what, device_name):
    """Turn a True/False/"auto" request into an on/off decision, warning when a request can't be honored."""
    if request == "auto":
        return bool(supported)
    if request and not supported:
        warnings.warn(f"{what} requested but not supported on {device_name}; falling back", stacklevel=3)
        return False
    return bool(request)


def configure_tf32(request="auto", device=None):
    """
    Enable or disable TF32 matmuls/convolutions for neural networks. `request` is True, False or
    "auto" (on if supported). Returns whether TF32 is now enabled. False explicitly disables it,
    so that "off" means full fp32.
    """
    caps = device_capabilities(device)
    name = f"{caps['device']} (compute capability {caps['compute_capability']})"
    enable = resolve(request, caps["tf32"], "TF32", name)
    torch.backends.cuda.matmul.allow_tf32 = enable
    torch.backends.cudnn.allow_tf32 = enable
    return enable
