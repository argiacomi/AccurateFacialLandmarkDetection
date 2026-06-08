"""Training runtime profiling helpers.

CUDA transfer/compute/eval sections use CUDA event timing when profiling is
enabled with ``--synchronize-runtime-timing``. That mode is enabled by default in
both the trainer and pipeline because it measures elapsed GPU work rather than
CPU enqueue time. Use ``--no-synchronize-runtime-timing`` to switch back to
low-overhead CPU wall-clock timings.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
import typing as T
from pathlib import Path

import torch


PHASE_TIMING_KEYS: tuple[str, ...] = (
    "data_wait_seconds",
    "device_transfer_seconds",
    "forward_loss_seconds",
    "backward_seconds",
    "optimizer_step_seconds",
    "scaler_update_seconds",
    "eval_seconds",
    "ema_eval_seconds",
    "distributed_eval_wait_seconds",
    "checkpoint_seconds",
)

COMPUTE_COMPONENT_KEYS: tuple[str, ...] = (
    "forward_loss_seconds",
    "backward_seconds",
    "optimizer_step_seconds",
    "scaler_update_seconds",
)


@dataclass
class TimingStart:
    wall_start: float
    device: torch.device | int | None = None
    start_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None

    @property
    def uses_cuda_events(self) -> bool:
        return self.start_event is not None and self.end_event is not None


def _is_cuda_timing_device(device: torch.device | int | str | None) -> bool:
    if device is None or not torch.cuda.is_available():
        return False
    if isinstance(device, int):
        return True
    try:
        return torch.device(device).type == "cuda"
    except (TypeError, RuntimeError):
        return False


def _maybe_synchronize(device: torch.device | int | None, synchronize: bool) -> None:
    if synchronize and _is_cuda_timing_device(device):
        torch.cuda.synchronize(device)


def cuda_peak_memory_mb(device: torch.device | int | None) -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        return round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 3)
    except Exception:
        return None


def runtime_metrics_path(args: T.Any) -> Path:
    if getattr(args, "runtime_metrics_jsonl", ""):
        return Path(args.runtime_metrics_jsonl)
    return Path(args.ckpt_folder) / "runtime_metrics.jsonl"


def append_runtime_metrics(args: T.Any, payload: T.Mapping[str, T.Any]) -> None:
    path = runtime_metrics_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def empty_epoch_timing() -> dict[str, float]:
    return {key: 0.0 for key in PHASE_TIMING_KEYS}


def start_timing(*, device: torch.device | int | None = None, synchronize: bool = False) -> TimingStart:
    """Start a timed section.

    With a CUDA device and synchronize=True, this records a CUDA event on the
    current stream. ``elapsed_timing`` then records and synchronizes an end event
    and returns GPU elapsed time in seconds. Without CUDA event timing, this
    falls back to CPU wall-clock timing.
    """

    if synchronize and _is_cuda_timing_device(device):
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return TimingStart(
            wall_start=time.time(),
            device=device,
            start_event=start_event,
            end_event=end_event,
        )
    return TimingStart(wall_start=time.time(), device=device)


def elapsed_timing(
    started_at: TimingStart | float,
    *,
    device: torch.device | int | None = None,
    synchronize: bool = False,
) -> float:
    if isinstance(started_at, TimingStart):
        if started_at.uses_cuda_events:
            assert started_at.start_event is not None
            assert started_at.end_event is not None
            started_at.end_event.record()
            started_at.end_event.synchronize()
            return float(started_at.start_event.elapsed_time(started_at.end_event)) / 1000.0
        _maybe_synchronize(started_at.device if device is None else device, synchronize)
        return time.time() - started_at.wall_start

    _maybe_synchronize(device, synchronize)
    return time.time() - float(started_at)


def accumulate_timing(
    timing: dict[str, float],
    key: str,
    started_at: TimingStart | float,
    *,
    device: torch.device | int | None = None,
    synchronize: bool = False,
) -> None:
    timing[key] = float(timing.get(key, 0.0)) + elapsed_timing(
        started_at,
        device=device,
        synchronize=synchronize,
    )


def time_call(timing: dict[str, float] | None, key: str, fn: T.Callable[..., T.Any], *args: T.Any, **kwargs: T.Any) -> T.Any:
    started_at = time.time()
    try:
        return fn(*args, **kwargs)
    finally:
        if timing is not None:
            timing[key] = float(timing.get(key, 0.0)) + (time.time() - started_at)


def finalize_epoch_timing(timing: dict[str, float], epoch_wall_seconds: float) -> dict[str, float]:
    final = dict(timing)
    compute_seconds = sum(float(final.get(key, 0.0)) for key in COMPUTE_COMPONENT_KEYS)
    final["compute_seconds"] = compute_seconds

    # Backward-compatible aggregate for older runtime_metrics.jsonl consumers.
    final["forward_backward_update_seconds"] = compute_seconds

    final["epoch_wall_seconds"] = float(epoch_wall_seconds)
    final["unattributed_seconds"] = max(
        0.0,
        final["epoch_wall_seconds"] - sum(float(final.get(key, 0.0)) for key in PHASE_TIMING_KEYS),
    )
    return final
