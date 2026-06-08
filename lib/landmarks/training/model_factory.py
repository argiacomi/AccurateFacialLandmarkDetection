from __future__ import annotations

import typing as T

from lib.landmarks.core.schema import DEFAULT_SCHEMA_HEADS
from lib.landmarks.models.attention import SA2SA1_2
from lib.landmarks.models.cdvit import HeadingNet, VitAttnStage


def build_backbone_net(heatmap_size: int) -> tuple[T.Callable[[int], HeadingNet], int]:
    heatmap_size = int(heatmap_size)
    if heatmap_size == 8:
        return lambda max_depth: HeadingNet([32, 64, 128, 128, max_depth]), 1
    if heatmap_size == 16:
        return lambda max_depth: HeadingNet([32, 64, 128, max_depth]), 1
    if heatmap_size == 32:
        return lambda max_depth: HeadingNet([32, 64, max_depth]), 2
    if heatmap_size == 64:
        return lambda max_depth: HeadingNet([32, max_depth]), 2
    raise ValueError("--heatmap_size must be one of 8, 16, 32, or 64")


def build_cdvit_model(
    args: T.Any,
    lmk_num: int,
    *,
    schema_aware_training: bool,
    auxiliary_class_names: T.Mapping[str, tuple[str, ...]],
) -> VitAttnStage:
    backbone_net, win_size = build_backbone_net(int(args.heatmap_size))
    return VitAttnStage(
        lmk_num=int(lmk_num),
        nstack=args.nstack,
        Attn=lambda: SA2SA1_2(args.heatmap_size, args.max_depth, win_size=win_size),
        heatmap_size=args.heatmap_size,
        max_depth=args.max_depth,
        backbone_net=backbone_net,
        schema_heads=DEFAULT_SCHEMA_HEADS if schema_aware_training else None,
        auxiliary_heads={name: len(labels) for name, labels in auxiliary_class_names.items()}
        if schema_aware_training and args.auxiliary_heads
        else None,
        visibility_heads=schema_aware_training and bool(getattr(args, "visibility_heads", True)),
    )
