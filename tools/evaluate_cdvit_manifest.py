#!/usr/bin/env python3
"""Evaluate a CD-ViT landmark checkpoint on an FS68 manifest once and exit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.core.schema import DEFAULT_SCHEMA_HEADS
from lib.datasets.registry import GetDataset
from lib.evaluation.split_safe import (
    EVAL_MODES,
    SPLIT_POLICIES,
    validate_no_train_test_leakage,
    write_eval_csv,
    write_eval_json,
    write_eval_records_csv,
    write_eval_records_jsonl,
)
from lib.models.attention import SA2SA1_2
from lib.models.cdvit import HeadingNet, VitAttnStage
from lib.training.evaluator import (
    _eval_collate,
    _evaluate_landmark_model,
    _print_eval_summary,
)
from lib.training.heatmap_stage import FS68_DATASET_NAME


def _backbone_for_heatmap_size(heatmap_size: int, max_depth: int):
    if heatmap_size == 8:
        return lambda max_depth: HeadingNet([32, 64, 128, 128, max_depth]), 1
    if heatmap_size == 16:
        return lambda max_depth: HeadingNet([32, 64, 128, max_depth]), 1
    if heatmap_size == 32:
        return lambda max_depth: HeadingNet([32, 64, max_depth]), 2
    if heatmap_size == 64:
        return lambda max_depth: HeadingNet([32, max_depth]), 2
    raise ValueError("--heatmap-size must be one of 8, 16, 32, or 64")


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint {path} did not contain a state dict")
    return {key.removeprefix("module."): value for key, value in state.items()}


def _checkpoint_has_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> bool:
    return any(str(key).startswith(prefix) for key in state_dict)


def _checkpoint_has_schema_heads(state_dict: dict[str, torch.Tensor]) -> bool:
    return _checkpoint_has_prefix(state_dict, "schema_output_layers.")


def _checkpoint_has_visibility_heads(state_dict: dict[str, torch.Tensor]) -> bool:
    return _checkpoint_has_prefix(state_dict, "visibility_output_layers.")


def _resolve_checkpoint_model_features(
    args: argparse.Namespace,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Enable schema/visibility model features needed by the checkpoint.

    Standalone eval historically instantiated schema heads only when the user
    passed --schema-aware-model. Visibility-capable checkpoints add
    visibility_output_layers.* keys, so eval must instantiate those modules
    before loading the checkpoint.

    CLI behavior:
      --visibility-heads     force visibility heads on
      --no-visibility-heads  force visibility heads off
      omitted                auto-enable if checkpoint has visibility heads
    """

    checkpoint_has_schema = _checkpoint_has_schema_heads(state_dict)
    checkpoint_has_visibility = _checkpoint_has_visibility_heads(state_dict)

    if getattr(args, "visibility_heads", None) is None:
        args.visibility_heads = checkpoint_has_visibility

    # Visibility heads are schema-specific, so a visibility-capable checkpoint
    # requires schema-aware construction too.
    if checkpoint_has_schema or bool(args.visibility_heads):
        args.schema_aware_model = True


def _build_model(
    args: argparse.Namespace,
    device: torch.device,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> VitAttnStage:
    if state_dict is None:
        state_dict = _load_state_dict(args.checkpoint, device)
        _resolve_checkpoint_model_features(args, state_dict)

    backbone_net, win_size = _backbone_for_heatmap_size(
        args.heatmap_size, args.max_depth
    )
    model = VitAttnStage(
        lmk_num=68,
        nstack=args.nstack,
        Attn=lambda: SA2SA1_2(args.heatmap_size, args.max_depth, win_size=win_size),
        heatmap_size=args.heatmap_size,
        max_depth=args.max_depth,
        backbone_net=backbone_net,
        schema_heads=DEFAULT_SCHEMA_HEADS if args.schema_aware_model else None,
        visibility_heads=bool(getattr(args, "visibility_heads", False)),
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"checkpoint missing keys ignored: {len(missing)}")
    if unexpected:
        print(f"checkpoint unexpected keys ignored: {len(unexpected)}")

    print(
        "standalone eval model features: "
        f"schema_aware_model={bool(args.schema_aware_model)} "
        f"schema_aware_eval={bool(args.schema_aware_eval)} "
        f"visibility_heads={bool(getattr(args, 'visibility_heads', False))}"
    )
    return model


def _dataset(args: argparse.Namespace, split: str):
    return GetDataset(
        FS68_DATASET_NAME,
        args.manifest,
        split,
        preload=args.preload != 0,
        aug=False,
        heatmap_size=0,
        manifest_path=args.manifest,
        eval_mode=args.eval_mode,
        heldout_datasets=args.heldout_dataset,
        include_metadata=split == "test",
        schema_aware_training=bool(args.schema_aware_eval),
        split_policy=args.split_policy,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval-mode", choices=EVAL_MODES, default="random_hash")
    parser.add_argument(
        "--split-policy", choices=SPLIT_POLICIES, default="declared_or_random_hash"
    )
    parser.add_argument(
        "--respect-declared-splits",
        action="store_true",
        help="Alias for --split-policy declared.",
    )
    parser.add_argument(
        "--ignore-declared-splits",
        action="store_true",
        help="Alias for --split-policy random_hash.",
    )
    parser.add_argument(
        "--heldout-dataset",
        action="append",
        default=[],
        help="Dataset label to hold out. by_dataset accepts one or more; leave_one_dataset_out requires exactly one.",
    )
    parser.add_argument("--eval-report-json", required=True)
    parser.add_argument("--eval-report-csv", default="")
    parser.add_argument("--eval-records-jsonl", default="")
    parser.add_argument("--eval-records-csv", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preload", type=int, default=1)
    parser.add_argument("--nstack", type=int, default=8)
    parser.add_argument("--heatmap-size", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=256)
    parser.add_argument("--schema-aware-model", action="store_true")
    parser.add_argument(
        "--schema-aware-eval",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Evaluate manifest samples through native schema heads. Defaults to --schema-aware-model.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.respect_declared_splits and args.ignore_declared_splits:
        parser.error(
            "pass only one of --respect-declared-splits or --ignore-declared-splits"
        )
    if args.respect_declared_splits:
        args.split_policy = "declared"
    if args.ignore_declared_splits:
        args.split_policy = "random_hash"
    device = torch.device(args.device)
    checkpoint_state_dict = _load_state_dict(args.checkpoint, device)
    _resolve_checkpoint_model_features(args, checkpoint_state_dict)

    if args.schema_aware_eval is None:
        args.schema_aware_eval = bool(args.schema_aware_model)

    train_dataset = _dataset(args, "train")
    test_dataset = _dataset(args, "test")
    validate_no_train_test_leakage(train_dataset.samples, test_dataset.samples)
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_eval_collate,
    )

    model = _build_model(args, device, checkpoint_state_dict)
    with torch.no_grad():
        report = _evaluate_landmark_model(
            model,
            dataloader,
            device,
            include_records=bool(args.eval_records_jsonl or args.eval_records_csv),
        )
    _print_eval_summary("checkpoint eval", report)
    records = report.pop("records", [])
    payload = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "eval_mode": args.eval_mode,
        "split_policy": args.split_policy,
        "heldout_datasets": list(args.heldout_dataset),
        "model": report,
    }
    write_eval_json(args.eval_report_json, payload)
    if args.eval_report_csv:
        write_eval_csv(args.eval_report_csv, payload)
    if args.eval_records_jsonl:
        write_eval_records_jsonl(args.eval_records_jsonl, records)
    if args.eval_records_csv:
        write_eval_records_csv(args.eval_records_csv, records)
    print(
        json.dumps(
            {"eval_report_json": args.eval_report_json, "record_count": len(records)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
