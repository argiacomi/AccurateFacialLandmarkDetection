"""Guards for the staged visibility-head compute path.

The landmark-conditioned visibility head is the most expensive per-stage head
(dominated by its 1x1 feature projection), so by default it runs only on the
final stack stage -- the only stage the loss and evaluator consume unless
``auxiliary_loss_stage == "all"``. Non-final visibility modules are still
instantiated (for checkpoint compatibility) but their forward is skipped, which
deliberately leaves their parameters unused in the default schema-aware path.

These tests pin two contracts:

1. The per-stage visibility gating (non-DDP, fast, always runs).
2. That a single-process DDP wrap with ``find_unused_parameters=True`` can run
   forward + loss on the final visibility output and backward without raising
   on the now-unused non-final visibility parameters. This guards against future
   changes to ``find_unused_parameters``, the model-factory default, or the
   visibility staging breaking the default training path.
"""

import os

import pytest
import torch
import torch.nn as nn

from lib.models.cdvit import VitAttnStage


def _tiny_visibility_model(nstack, visibility_all_stages):
    """A minimal multi-schema model with visibility heads and >1 stack.

    Mirrors the lightweight construction used elsewhere in the suite: tiny
    heatmap size, identity attention, and a trivial conv backbone so the test
    exercises the staging/DDP wiring rather than real backbone compute.
    """
    return VitAttnStage(
        lmk_num=68,
        nstack=nstack,
        heatmap_size=8,
        max_depth=16,
        backbone_net=lambda max_depth: nn.Sequential(
            nn.Conv2d(3, max_depth, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((8, 8)),
        ),
        Attn=lambda: nn.Identity(),
        num_dvit_per_pred_blk=1,
        schema_heads={"landmarks_68": 68, "landmarks_98": 98},
        visibility_heads=True,
        visibility_all_stages=visibility_all_stages,
    )


def _visibility_keys(stage_output):
    return sorted(k for k in stage_output if k.startswith("visibility_"))


def test_visibility_runs_only_on_final_stage_by_default():
    model = _tiny_visibility_model(nstack=3, visibility_all_stages=False).eval()
    with torch.no_grad():
        stages = model(torch.zeros(2, 3, 32, 32))

    # Non-final stages omit visibility outputs entirely.
    for stage in stages[:-1]:
        assert _visibility_keys(stage) == []
    # Final stage carries the per-schema visibility heads.
    assert _visibility_keys(stages[-1]) == ["visibility_68", "visibility_98"]

    # Landmark outputs remain on every stage regardless of visibility staging.
    for stage in stages:
        assert "landmarks_68" in stage and "landmarks_98" in stage


def test_visibility_runs_on_every_stage_when_all_stages_enabled():
    model = _tiny_visibility_model(nstack=3, visibility_all_stages=True).eval()
    with torch.no_grad():
        stages = model(torch.zeros(2, 3, 32, 32))

    for stage in stages:
        assert _visibility_keys(stage) == ["visibility_68", "visibility_98"]


def test_non_final_visibility_modules_are_instantiated_but_unused():
    """Modules exist for every stage (checkpoint-safe) but only the final one
    contributes gradients in the default path."""
    model = _tiny_visibility_model(nstack=3, visibility_all_stages=False)

    # All stages keep their visibility modules so checkpoints stay compatible.
    for layers in model.visibility_output_layers.values():
        assert len(layers) == len(model.stages)

    model.train()
    stages = model(torch.zeros(2, 3, 32, 32))
    final = stages[-1]
    loss = final["visibility_68"].pow(2).mean() + final["landmarks_68"][0].pow(2).mean()
    loss.backward()

    # Non-final visibility parameters receive no gradient; the final ones do.
    unused = {
        name
        for name, param in model.named_parameters()
        if param.requires_grad and "visibility" in name and param.grad is None
    }
    used = {
        name
        for name, param in model.named_parameters()
        if param.requires_grad and "visibility" in name and param.grad is not None
    }
    assert unused, "expected non-final visibility params to be unused"
    assert used, "expected final-stage visibility params to receive gradient"


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed with Gloo backend is required for the DDP smoke test",
)
def test_ddp_backward_tolerates_unused_non_final_visibility_params():
    """Single-process DDP (CPU/Gloo) must run forward + final visibility loss +
    backward without raising on the unused non-final visibility parameters.

    This mirrors the schema-aware training wrap, which sets
    ``find_unused_parameters=True``.
    """
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    already_initialized = dist.is_initialized()
    if not already_initialized:
        try:
            dist.init_process_group(
                backend="gloo", rank=0, world_size=1, init_method="env://"
            )
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - env dependent
            pytest.skip(f"could not initialize Gloo process group: {exc}")

    try:
        model = _tiny_visibility_model(nstack=3, visibility_all_stages=False)
        ddp_model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=None,  # CPU
            find_unused_parameters=True,
        )
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=0.0, weight_decay=1e-3)

        # Two iterations are required for this to be a real guard: the DDP
        # reducer only detects an unused parameter from the *previous* step, so
        # a single backward would pass even with find_unused_parameters=False.
        # With it False, the second backward raises "Expected to have finished
        # reduction in the prior iteration"; with it True (the schema-aware
        # default), both iterations succeed.
        for _ in range(2):
            optimizer.zero_grad()
            stages = ddp_model(torch.zeros(2, 3, 32, 32))
            final = stages[-1]
            # Loss on the final-stage visibility output (and a landmark term) only.
            loss = (
                final["visibility_68"].pow(2).mean()
                + final["visibility_98"].pow(2).mean()
                + final["landmarks_68"][0].pow(2).mean()
            )
            loss.backward()  # must not raise on unused non-final visibility params
            optimizer.step()

        # The reducer accepted the unused params; confirm the final-stage
        # visibility head still learned (received gradient).
        final_visibility_grad = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for name, param in model.named_parameters()
            if "visibility" in name and param.grad is not None
        )
        assert final_visibility_grad
    finally:
        if not already_initialized and dist.is_initialized():
            dist.destroy_process_group()
