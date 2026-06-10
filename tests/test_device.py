"""Tests for accelerator selection and compile/AMP helpers.

These run on whatever backend the host provides (CUDA, MPS, or CPU); assertions
are written against ``torch``'s reported availability so they hold everywhere.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from lib.training.ddp import LocalModelWrapper
from lib.training.device import (
    autocast,
    compile_model,
    default_device_str,
    make_grad_scaler,
    resolve_device,
    select_compile_backend,
    supports_amp,
)


def test_default_device_str_prefers_available_accelerator():
    expected = "cpu"
    if torch.cuda.is_available():
        expected = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        expected = "mps"
    assert default_device_str() == expected


def test_resolve_device_auto_matches_default():
    assert resolve_device("auto").type == torch.device(default_device_str()).type
    assert resolve_device(None).type == torch.device(default_device_str()).type


def test_resolve_device_cpu_always_available():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_unavailable_backend_raises():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("mps")


def test_grad_scaler_enabled_only_on_cuda():
    assert make_grad_scaler(torch.device("cpu")).is_enabled() is False
    device = resolve_device("auto")
    assert make_grad_scaler(device).is_enabled() is supports_amp(device)


def test_autocast_is_noop_off_cuda():
    # CPU/MPS keep fp32: the helper returns a plain null context, not autocast.
    assert isinstance(autocast(torch.device("cpu")), contextlib.nullcontext)


def test_local_model_wrapper_mirrors_ddp_surface():
    base = torch.nn.Linear(4, 2)
    wrapped = LocalModelWrapper(base)

    assert wrapped.module is base
    # Parameters are shared (not copied) so an optimizer sees the real weights.
    assert next(wrapped.parameters()) is next(base.parameters())
    # state_dict keys are prefix-free, matching an unwrapped/eager checkpoint.
    assert list(wrapped.module.state_dict().keys()) == list(base.state_dict().keys())

    out = wrapped(torch.randn(3, 4))
    assert tuple(out.shape) == (3, 2)


def test_select_compile_backend_auto_is_device_aware():
    # Inductor everywhere except MPS, where its backward codegen is unstable.
    assert select_compile_backend(torch.device("cuda"), "auto") == "inductor"
    assert select_compile_backend(torch.device("cpu"), "auto") == "inductor"
    assert select_compile_backend(None, "auto") == "inductor"
    assert select_compile_backend(torch.device("mps"), "auto") == "aot_eager"


def test_select_compile_backend_explicit_is_honored():
    assert select_compile_backend(torch.device("mps"), "inductor") == "inductor"
    assert select_compile_backend(torch.device("cuda"), "aot_eager") == "aot_eager"


def test_compile_model_preserves_module_and_state_dict_keys():
    base = torch.nn.Linear(4, 2)
    base_keys = list(base.state_dict().keys())

    compiled = compile_model(LocalModelWrapper(base))

    # torch.compile is lazy, so this exercises wrapping/delegation without
    # triggering a backend compilation (no compiler toolchain required).
    assert type(compiled).__name__ == "OptimizedModule"
    assert compiled.module is base
    assert list(compiled.module.state_dict().keys()) == base_keys
