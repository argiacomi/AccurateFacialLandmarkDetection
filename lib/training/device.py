"""Accelerator (CUDA / MPS / CPU) selection and AMP helpers.

Centralizes device resolution so the trainer, evaluator, and tools run on an
NVIDIA GPU (CUDA), Apple Silicon (MPS), or CPU from the same code paths.

Automatic mixed precision (fp16 autocast + GradScaler) and forced FlashAttention
are CUDA-only. On MPS/CPU these helpers return no-op contexts and a disabled
GradScaler so the model runs in fp32, which is the supported configuration on
those backends.
"""

from __future__ import annotations

import contextlib

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def mps_available() -> bool:
    """Whether the Apple Silicon Metal (MPS) backend is usable."""

    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def default_device_str() -> str:
    """Return the best available device string: cuda, then mps, then cpu."""

    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def resolve_device(preference: str | torch.device | None = "auto") -> torch.device:
    """Resolve a device preference to a concrete, available ``torch.device``.

    ``"auto"`` (or ``None``) picks the best available accelerator. An explicit
    request for an unavailable backend raises so a misconfigured run fails
    loudly instead of silently falling back to CPU.
    """

    if preference is None or preference == "auto":
        return torch.device(default_device_str())

    device = torch.device(preference)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is False"
        )
    if device.type == "mps" and not mps_available():
        raise RuntimeError(
            "MPS device requested but torch.backends.mps.is_available() is False"
        )
    return device


def supports_amp(device: torch.device) -> bool:
    """Whether fp16 autocast + GradScaler should be enabled for ``device``."""

    return device.type == "cuda"


def autocast(device: torch.device, dtype: torch.dtype = torch.float16):
    """Return an fp16 autocast context on CUDA, else a no-op context.

    GradScaler-backed fp16 AMP is only reliable on CUDA. On MPS/CPU the forward
    pass stays in fp32 to avoid unsupported or unstable half-precision kernels.
    """

    if supports_amp(device):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def make_grad_scaler(device: torch.device) -> torch.amp.GradScaler:
    """Return a GradScaler enabled only on CUDA.

    When disabled the scaler is a transparent pass-through: ``scale`` returns the
    loss unchanged, ``step`` calls ``optimizer.step`` directly, and ``update`` is
    a no-op, so the identical training loop runs on MPS/CPU in fp32.
    """

    enabled = supports_amp(device)
    return torch.amp.GradScaler("cuda" if enabled else "cpu", enabled=enabled)


def attention_kernel(device: torch.device):
    """Return the scaled-dot-product-attention kernel context for ``device``.

    FlashAttention is CUDA-only; forcing it on other backends makes
    ``scaled_dot_product_attention`` raise. On MPS/CPU we let PyTorch select a
    supported kernel.
    """

    if device.type == "cuda":
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    return contextlib.nullcontext()


__all__ = [
    "attention_kernel",
    "autocast",
    "default_device_str",
    "make_grad_scaler",
    "mps_available",
    "resolve_device",
    "supports_amp",
]
