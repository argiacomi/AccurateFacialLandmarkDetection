"""CD-ViT heatmap training implementation.

This module backs the legacy TrainHeatmapStageFP16.py entrypoint.  Keep
long-lived training logic here and keep TrainHeatmapStageFP16.py as a thin
compatibility wrapper for existing commands, pipeline invocations, and
checkpoint metadata.
"""

# ruff: noqa: E402
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_CDVIT_ROOT = _Path(__file__).resolve().parents[3]
if str(_CDVIT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CDVIT_ROOT))

import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

# from Vit import Vit
# from Attention import  SA2SA1_twins
# from UNet2 import UNet
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from lib.core.manifest_aliases import (
    CANONICAL_MANIFEST_DATA_NAME,
    LEGACY_MANIFEST_DATA_NAME,
    is_schema_aware_manifest_dataset,
)
from lib.evaluation.split_safe import (
    write_eval_csv,
    write_eval_json,
    write_eval_records_csv,
    write_eval_records_jsonl,
)
from lib.logging_utils import (
    Verbosity,
    configure_console_logging,
    fmt_count,
    fmt_duration,
    fmt_mapping,
    fmt_num,
    fmt_progress,
    is_verbose,
    log_event,
    start_training_progress,
    stop_training_progress,
    summarize_mapping,
    update_training_progress,
    verbosity_from_name,
)
from lib.training.checkpoint_compat import (
    checkpoint_compat_errors_from_args as _checkpoint_compat_errors,
)
from lib.training.checkpointing import (
    _collect_rng_state_by_rank,
    _restore_training_checkpoint,
    _save_training_checkpoint,
    _set_checkpoint_rng_state_by_rank,
    _torch_load_training_checkpoint,
    _write_training_complete_sentinel,
)
from lib.training.cli import build_heatmap_stage_arg_parser
from lib.training.config import (
    DatasetBuildConfig,
    EvalConfig,
    TrainingRuntimeConfig,
    config_dict,
)
from lib.training.data import (
    AUXILIARY_CLASS_NAMES,
    batch_mix,
    build_dataset,
    landmark_count_for_dataset,
    schema_aware_collate,
    unpack_train_batch,
)
from lib.training.ddp import (
    LocalModelWrapper,
    distributed_is_active,
    distributed_rank,
    distributed_world_size,
    is_rank_zero,
    setup_distributed_from_env,
)
from lib.training.device import (
    attention_kernel,
    autocast,
    compile_model,
    make_grad_scaler,
    select_compile_backend,
)
from lib.training.ema import EMA
from lib.training.eval_schedule import build_eval_schedule
from lib.training.evaluator import (
    eval_collate,
    eval_report_json_path,
    evaluate_landmark_model,
    print_eval_summary,
    records_from_report,
)
from lib.training.loaders import build_training_loaders
from lib.training.losses import (
    heatmap_batch_weight,
    schema_head_loss,
    weighted_smooth_l1,
)
from lib.training.model_factory import build_cdvit_model
from lib.training.profiling import (
    accumulate_timing,
    append_runtime_metrics,
    cuda_peak_memory_mb,
    elapsed_timing,
    empty_epoch_timing,
    finalize_epoch_timing,
    start_timing,
    time_call,
)
from lib.training.runtime import (
    dataloader_kwargs,
    maybe_limit_eval_dataset,
    normalize_runtime_args,
    set_dataset_runtime_epoch,
    should_run_interval,
)
from lib.training.seed import setup_seed
from loss import AWingLoss, STARLoss_v2

# from torch.cuda.amp import autocast as autocast


LEGACY_FS68_DATASET_NAME = LEGACY_MANIFEST_DATA_NAME
MULTI_SCHEMA_MANIFEST_DATASET_NAME = CANONICAL_MANIFEST_DATA_NAME
FS68_DATASET_NAME = LEGACY_FS68_DATASET_NAME

# With AMP enabled a non-finite loss can be a one-off fp16 overflow that
# GradScaler absorbs by skipping the step and shrinking the loss scale; only
# this many consecutive non-finite batches abort training as real divergence.
MAX_NONFINITE_LOSS_STREAK = 5

# Legacy private helper aliases kept for TrainHeatmapStageFP16.py and older tests/tools.
_landmark_count_for_dataset = landmark_count_for_dataset
_build_dataset = build_dataset
_batch_mix = batch_mix
_schema_aware_collate = schema_aware_collate
_unpack_train_batch = unpack_train_batch
_eval_collate = eval_collate
_eval_report_json_path = eval_report_json_path
_evaluate_landmark_model = evaluate_landmark_model
_print_eval_summary = print_eval_summary
_records_from_report = records_from_report
_weighted_smooth_l1 = weighted_smooth_l1
_schema_head_loss = schema_head_loss
_heatmap_batch_weight = heatmap_batch_weight
_build_cdvit_model = build_cdvit_model
_dataloader_kwargs = dataloader_kwargs
_maybe_limit_eval_dataset = maybe_limit_eval_dataset
_normalize_runtime_args = normalize_runtime_args
_set_dataset_runtime_epoch = set_dataset_runtime_epoch
_should_run_interval = should_run_interval
_accumulate_timing = accumulate_timing
_append_runtime_metrics = append_runtime_metrics
_cuda_peak_memory_mb = cuda_peak_memory_mb
_elapsed_timing = elapsed_timing
_empty_epoch_timing = empty_epoch_timing
_finalize_epoch_timing = finalize_epoch_timing
_start_timing = start_timing
_time_call = time_call
_build_training_loaders = build_training_loaders
_build_eval_schedule = build_eval_schedule
_dataset_build_config = DatasetBuildConfig.from_args
_eval_config = EvalConfig.from_args
_training_runtime_config = TrainingRuntimeConfig.from_args


# Runtime, profiling, and checkpoint-compat helpers live in lib.landmarks.training.*.


# Checkpoint save/load/RNG helpers live in lib.landmarks.training.checkpointing.


# Stable column order for the per-batch training loss breakdown.
_LOSS_COMPONENT_ORDER = ("loc", "heat", "star", "cons", "vis", "aux")


def _log_schema_head_details(loss_details) -> None:
    """Log the full schema-head diagnostics line (``--log-level verbose``).

    The per-head counts/contributions/aux/visibility breakdown is wide and hard
    to compare across steps, so it is verbose-only; the caller gates on
    :func:`is_verbose` to also skip the per-head ``.item()`` syncs otherwise. The
    complete payload is always retained in runtime_metrics.jsonl.
    """

    head_counts = loss_details.get("head_sample_counts", {})
    head_losses = {
        name: float(value.item())
        for name, value in loss_details.get("head_loss_contributions", {}).items()
    }
    vis_weight = float(
        loss_details.get("visibility_loss_weight", torch.tensor(0.0)).item()
    )
    log_event(
        "train",
        "  heads | "
        f"counts {fmt_mapping(head_counts)} | "
        f"contrib {fmt_mapping(head_losses)} | "
        f"aux_valid {fmt_mapping(loss_details.get('auxiliary_valid_counts', {}))} | "
        f"aux_acc {fmt_mapping(loss_details.get('auxiliary_accuracy', {}))} | "
        f"vis_valid {fmt_mapping(loss_details.get('visibility_valid_counts', {}))} | "
        f"vis_w {fmt_num(vis_weight)}",
        level=Verbosity.VERBOSE,
    )


def _save_best_weights(state_dict, ckpt_folder: str | os.PathLike[str]) -> None:
    """Write the best model weights."""

    ckpt_path = Path(ckpt_folder)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, ckpt_path / "best.weights.pt")


def _distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _is_schema_extension_key(key: str) -> bool:
    return key.startswith(("schema_output_layers.", "auxiliary_output_layers."))


def _load_resume_model_state(net, state_dict, args) -> None:
    if not getattr(args, "allow_missing_schema_heads", False):
        net.load_state_dict(state_dict)
        return

    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    bad_missing = [key for key in missing if not _is_schema_extension_key(key)]
    if bad_missing or unexpected:
        raise ValueError(
            "checkpoint is not compatible with --allow-missing-schema-heads: "
            f"unexpected={unexpected[:10]!r} non_schema_missing={bad_missing[:10]!r}"
        )
    if missing:
        log_event(
            "resume",
            f"{len(missing)} schema/auxiliary head key(s) missing from "
            f"checkpoint; initialized from current model ({missing[:10]})",
            level=Verbosity.VERBOSE,
        )
    if unexpected:
        log_event(
            "resume",
            f"{len(unexpected)} unexpected checkpoint key(s) ignored "
            f"({unexpected[:10]})",
            level=Verbosity.VERBOSE,
        )


def _merge_count_dict(target: dict, source: dict) -> None:
    for key, value in source.items():
        target[key] = int(target.get(key, 0)) + int(value)


def _aggregate_sampler_diagnostics(local_diagnostics):
    """Aggregate domain-balanced sampler diagnostics across DDP ranks.

    `DomainBalancedBatchSampler.last_epoch_diagnostics` is rank-local. For useful
    epoch-level reporting, gather diagnostics from every rank and sum actual mix
    counts and fallback counts before rank 0 logs runtime metrics.
    """

    if not (dist.is_available() and dist.is_initialized()):
        return local_diagnostics

    world_size = distributed_world_size()
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_diagnostics or {})

    if not is_rank_zero():
        return local_diagnostics

    requested_targets = {}
    actual_mix = {"bucket": {}, "dataset": {}, "schema": {}}
    fallback_counts = {}
    missing_targets = {"bucket": set(), "dataset": set(), "schema": set()}
    rank_diagnostics = []

    for rank, diagnostics in enumerate(gathered):
        if not isinstance(diagnostics, dict):
            continue

        if not requested_targets:
            requested_targets = dict(diagnostics.get("requested_targets", {}))

        for category, counts in dict(diagnostics.get("actual_mix", {})).items():
            actual_mix.setdefault(category, {})
            if isinstance(counts, dict):
                _merge_count_dict(actual_mix[category], counts)

        counts = diagnostics.get("fallback_counts", {})
        if isinstance(counts, dict):
            _merge_count_dict(fallback_counts, counts)

        missing = diagnostics.get("missing_targets", {})
        if isinstance(missing, dict):
            for category, values in missing.items():
                missing_targets.setdefault(category, set())
                for value in values or ():
                    missing_targets[category].add(str(value))

        rank_diagnostics.append(
            {
                "rank": int(diagnostics.get("rank", rank)),
                "batches_per_rank": int(diagnostics.get("batches_per_rank", 0)),
            }
        )

    return {
        "requested_targets": requested_targets,
        "actual_mix": actual_mix,
        "fallback_counts": fallback_counts,
        "missing_targets": {
            category: sorted(values) for category, values in missing_targets.items()
        },
        "rank": "all",
        "world_size": int(world_size),
        "batches_per_rank": max(
            (item["batches_per_rank"] for item in rank_diagnostics),
            default=0,
        ),
        "rank_diagnostics": rank_diagnostics,
    }


def _sampler_summary_line(epoch: int, diagnostics: dict) -> str:
    """One-line domain-balanced sampler summary for the console.

    Renders the bucket mix as percentages plus fallback and missing-target
    counts. The full diagnostics object is preserved in runtime_metrics.jsonl
    and printed verbatim only under ``--log-level debug``.
    """

    actual_mix = diagnostics.get("actual_mix", {}) or {}
    bucket_mix = actual_mix.get("bucket", {}) or {}
    fallback_counts = diagnostics.get("fallback_counts", {}) or {}
    fallbacks = sum(
        int(count) for key, count in fallback_counts.items() if key != "exact"
    )
    missing_counts = {
        category: len(values or [])
        for category, values in (diagnostics.get("missing_targets", {}) or {}).items()
    }
    return (
        f"e{int(epoch):03d} domain mix | "
        f"bucket {summarize_mapping(bucket_mix, top_n=4, as_percent=True)} | "
        f"fallback {fallbacks} | "
        f"missing {fmt_mapping(missing_counts)}"
    )


def _batch_mix_summary_line(mix: dict) -> str:
    """Compact per-step batch mix summary.

    ``batch_mix`` returns nested count dictionaries. Logging the raw structure makes
    every train line hard to scan, so normal console output shows only top shares.
    The raw batch object can still be inspected through debug tooling if needed.
    """

    if not mix:
        return "-"
    parts: list[str] = []
    for category, top_n in (("bucket", 4), ("schema", 3), ("dataset", 3)):
        values = mix.get(category, {}) if isinstance(mix, dict) else {}
        if values:
            parts.append(
                f"{category} {summarize_mapping(values, top_n=top_n, as_percent=True)}"
            )
    return " | ".join(parts) if parts else "-"


def main():
    parser = build_heatmap_stage_arg_parser()
    args = parser.parse_args()
    if args.respect_declared_splits and args.ignore_declared_splits:
        parser.error(
            "pass only one of --respect-declared-splits or --ignore-declared-splits"
        )
    if args.respect_declared_splits:
        args.split_policy = "declared"
    if args.ignore_declared_splits:
        args.split_policy = "random_hash"
    args = normalize_runtime_args(args)
    configure_console_logging(
        verbosity_from_name(getattr(args, "log_level", "normal")),
        getattr(args, "log_format", "human"),
        configure_stdlib=False,
    )
    setup_seed(args.seed, deterministic=args.deterministic)
    lmk_num = landmark_count_for_dataset(args)
    device = setup_distributed_from_env(args)
    training_runtime_config = TrainingRuntimeConfig.from_args(args)
    eval_config = EvalConfig.from_args(args)
    dataset_build_config = DatasetBuildConfig.from_args(args)

    with attention_kernel(device):
        # if True:

        schema_aware_training = (
            is_schema_aware_manifest_dataset(args.data_name)
            and args.schema_aware_training
        )
        loaders = build_training_loaders(
            args,
            schema_aware_training=schema_aware_training,
            rank=distributed_rank(),
            world_size=distributed_world_size(),
        )
        train_dataset = loaders.train_dataset
        test_dataset = loaders.test_dataset
        eval_dataset = loaders.eval_dataset
        train_sampler = loaders.train_sampler
        train_dataloader = loaders.train_dataloader
        test_dataloader = loaders.test_dataloader
        full_test_dataloader = loaders.full_test_dataloader
        if is_rank_zero():
            log_event(
                "data",
                f"train {fmt_count(len(train_dataset))} samples | "
                f"test {fmt_count(len(test_dataset))} samples | device {device}",
                level=Verbosity.QUIET,
            )
        if is_rank_zero():
            append_runtime_metrics(
                args,
                {
                    "event": "config_snapshot",
                    "runtime": config_dict(training_runtime_config),
                    "eval": config_dict(eval_config),
                    "dataset": config_dict(dataset_build_config),
                },
            )
        net = build_cdvit_model(
            args,
            lmk_num,
            schema_aware_training=schema_aware_training,
            auxiliary_class_names=AUXILIARY_CLASS_NAMES,
        ).to(device)
        # net = VitAttnStage(
        #     nstack=args.nstack,
        #     Attn=lambda: SA2SA1_2(args.heatmap_size, args.max_depth),
        #     heatmap_size=args.heatmap_size,
        #     max_depth=args.max_depth,
        # ).cuda()
        # net = UNetStage(
        #     lmk_num=lmk_num,
        #     nstack=args.nstack,
        #     heatmap_size=args.heatmap_size,
        #     max_depth=args.max_depth,
        #     feature_extractor=Vit
        # ).cuda()
        resume_checkpoint = None
        start_epoch = 0
        if args.resume != "":
            resume_checkpoint = _torch_load_training_checkpoint(args.resume, device)
            if isinstance(resume_checkpoint, dict) and "model" in resume_checkpoint:
                compat_errors = _checkpoint_compat_errors(resume_checkpoint, args)
                if compat_errors and not args.allow_incompatible_resume:
                    raise ValueError(
                        "refusing to resume incompatible checkpoint: "
                        + "; ".join(compat_errors)
                        + ". Pass --allow-incompatible-resume only if this is intentional."
                    )
                _load_resume_model_state(net, resume_checkpoint["model"], args)
                start_epoch = int(
                    resume_checkpoint.get(
                        "next_epoch",
                        int(resume_checkpoint.get("epoch", -1)) + 1,
                    )
                )
            else:
                _load_resume_model_state(net, resume_checkpoint, args)
        ddp_find_unused = args.find_unused_parameters or schema_aware_training
        # Visibility modules are instantiated on every stage but, unless
        # auxiliary_loss_stage == "all", only the final stage runs. The
        # non-final visibility parameters are then unused in the backward pass,
        # so DDP must use find_unused_parameters=True. This is guaranteed today
        # because visibility heads require schema-aware training (see
        # model_factory), which forces the flag above; assert it so a future
        # change to that clause fails loudly instead of hanging DDP.
        visibility_heads_active = schema_aware_training and bool(
            getattr(args, "visibility_heads", True)
        )
        visibility_all_stages = (
            str(getattr(args, "auxiliary_loss_stage", "final")) == "all"
        )
        if visibility_heads_active and not visibility_all_stages:
            assert ddp_find_unused, (
                "Schema-aware visibility heads run only on the final stage "
                "(auxiliary_loss_stage != 'all'), leaving non-final visibility "
                "parameters unused; DDP requires find_unused_parameters=True. "
                "Pass --find_unused_parameters or --auxiliary-loss-stage all."
            )
            if is_rank_zero():
                log_event(
                    "ddp",
                    "visibility heads on final stage only; "
                    "find_unused_parameters=True (required for unused "
                    "non-final visibility parameters).",
                    level=Verbosity.VERBOSE,
                )
        if distributed_is_active():
            net = torch.nn.parallel.DistributedDataParallel(
                net,
                device_ids=[args.local_rank],
                find_unused_parameters=ddp_find_unused,
            )
        else:
            # Single-process (MPS/CPU or single-GPU without torchrun): expose the
            # same ``.module`` / call surface DDP would, without a process group.
            net = LocalModelWrapper(net)

        if getattr(args, "compile", False):
            # Compile outermost (after DDP) so net.module.state_dict() stays
            # prefix-free for checkpoints and Dynamo can split DDP graphs. The
            # uncompiled base model remains reachable via net.module, so EMA and
            # eval keep running in eager mode.
            compile_backend = select_compile_backend(
                device, getattr(args, "compile_backend", "auto")
            )
            net = compile_model(
                net,
                mode=getattr(args, "compile_mode", "default"),
                backend=getattr(args, "compile_backend", "auto"),
                device=device,
            )
            if is_rank_zero():
                log_event(
                    "compile",
                    f"enabled (backend={compile_backend}, "
                    f"mode={getattr(args, 'compile_mode', 'default')}); "
                    "expect extra warmup compilation on the first steps",
                    level=Verbosity.QUIET,
                )

        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = StepLR(optimizer, args.sched_step, gamma=0.5)

        best_nme = 99999
        weights = [1 / math.pow(args.mul, i) for i in range(args.nstack)]
        weights.reverse()
        best_record = []

        # heatmap_loss_func = HeatMapLoss2
        heatmap_loss_func = AWingLoss()
        vertex_loss_func = (
            STARLoss_v2(
                check_finite=bool(getattr(args, "star_loss_check_finite", False)),
                check_finite_interval=int(
                    getattr(args, "star_loss_check_finite_interval", 0)
                ),
            )
            if args.star_loss_weight > 0
            else None
        )
        ema = EMA(net.module, 0.99, 100, 10)
        scaler = make_grad_scaler(device)
        if isinstance(resume_checkpoint, dict) and "model" in resume_checkpoint:
            start_epoch, best_nme, best_record = _restore_training_checkpoint(
                resume_checkpoint,
                optimizer,
                scheduler,
                scaler,
                ema,
                best_nme,
                best_record,
                args,
            )
            if is_rank_zero():
                log_event(
                    "resume",
                    f"restored training checkpoint; resuming at epoch {start_epoch}",
                    level=Verbosity.QUIET,
                )
        if start_epoch >= args.epoch:
            if is_rank_zero():
                log_event(
                    "resume",
                    f"next_epoch={start_epoch} >= requested epoch="
                    f"{args.epoch}; training already complete for this target",
                    level=Verbosity.QUIET,
                )
            _distributed_barrier()
            return
        for epoch in range(start_epoch, args.epoch):
            args.current_epoch = int(epoch)
            n = 0
            net.train()
            ema.train()
            if is_rank_zero():
                epoch_start_time = time.time()
                epoch_timing = empty_epoch_timing()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)
            train_sampler.set_epoch(epoch)
            set_dataset_runtime_epoch(train_dataset, epoch, args)
            if args.persistent_workers and epoch == start_epoch and is_rank_zero():
                log_event(
                    "train",
                    "note: --persistent-workers seeds DataLoader workers "
                    "once for throughput; use --no-persistent-workers for "
                    "epoch-reseeded worker RNG",
                    level=Verbosity.VERBOSE,
                )
            total_train_steps = max(len(train_dataloader), 1)
            train_progress = start_training_progress(
                total_train_steps,
                description=f"e{int(epoch):03d}",
                enabled=is_rank_zero() and bool(getattr(args, "train_progress", True)),
            )
            nonfinite_loss_streak = 0
            batch_fetch_start_time = time.time()
            for batch_idx, batch in enumerate(train_dataloader):
                if is_rank_zero():
                    accumulate_timing(
                        epoch_timing, "data_wait_seconds", batch_fetch_start_time
                    )
                optimizer.zero_grad(set_to_none=True)
                schema_batch = isinstance(batch, dict)
                transfer_start_time = start_timing(
                    device=device, synchronize=args.synchronize_runtime_timing
                )
                if schema_batch:
                    data, schema_heads, aux_labels = unpack_train_batch(
                        batch, device, non_blocking=args.pin_memory
                    )
                else:
                    data, target, heatmap, sample_weight, landmark_mask = (
                        unpack_train_batch(batch, device, non_blocking=args.pin_memory)
                    )
                if is_rank_zero():
                    accumulate_timing(
                        epoch_timing,
                        "device_transfer_seconds",
                        transfer_start_time,
                        device=device,
                        synchronize=args.synchronize_runtime_timing,
                    )
                forward_loss_start_time = start_timing(
                    device=device, synchronize=args.synchronize_runtime_timing
                )
                loss = 0
                loss_loc = torch.tensor(0.0, device=device)
                loss_heatmap = torch.tensor(0.0, device=device)
                loss_aux = torch.tensor(0.0, device=device)
                loss_consistency = torch.tensor(0.0, device=device)
                loss_star = torch.tensor(0.0, device=device)
                loss_visibility = torch.tensor(0.0, device=device)
                loss_details = None
                # if True:
                with autocast(device):
                    pred_info = net(data)
                    for i in range(len(pred_info)):
                        if schema_batch:
                            (
                                stage_loss,
                                stage_loc,
                                stage_heatmap,
                                stage_aux,
                                stage_details,
                            ) = schema_head_loss(
                                pred_info[i],
                                schema_heads,
                                aux_labels,
                                heatmap_loss_func,
                                args,
                                return_details=True,
                                star_loss_func=vertex_loss_func,
                                include_auxiliary_loss=(
                                    getattr(args, "auxiliary_loss_stage", "final")
                                    == "all"
                                    or i == len(pred_info) - 1
                                ),
                                include_visibility_loss=(
                                    getattr(args, "auxiliary_loss_stage", "final")
                                    == "all"
                                    or i == len(pred_info) - 1
                                ),
                            )
                            loss_loc = stage_loc
                            loss_heatmap = stage_heatmap
                            loss_aux = stage_aux
                            loss_details = stage_details
                            loss_consistency = stage_details["loss_consistency"]
                            loss_star = stage_details["loss_star"]
                            loss_visibility = stage_details.get(
                                "loss_visibility", loss_visibility
                            )
                            loss = loss + stage_loss * weights[i]
                        else:
                            pred_loc, pred_heatmap = pred_info[i]
                            B, C, H, W = pred_heatmap.shape
                            # loss_loc = vertex_loss_func(pred_heatmap, target)
                            loss_loc = (
                                weighted_smooth_l1(
                                    pred_loc,
                                    target,
                                    sample_weight,
                                    landmark_mask,
                                    beta=0.001,
                                )
                                * args.locw
                            )
                            pred_prob = F.softmax(
                                pred_heatmap.reshape((B, C, -1)), dim=2
                            ).reshape((B, C, H, W))
                            loss_heatmap = (
                                heatmap_loss_func(
                                    pred_prob,
                                    heatmap,
                                    batch_weights=heatmap_batch_weight(
                                        sample_weight, pred_heatmap, landmark_mask
                                    ),
                                )
                                * args.hw
                            )  # for awing loss
                            # loss_heatmap = heatmap_loss_func(pred_heatmap, heatmap) * args.hw
                            loss = loss + (loss_loc + loss_heatmap) * weights[i]
                if is_rank_zero():
                    accumulate_timing(
                        epoch_timing,
                        "forward_loss_seconds",
                        forward_loss_start_time,
                        device=device,
                        synchronize=args.synchronize_runtime_timing,
                    )

                nonfinite_interval = int(
                    getattr(args, "nonfinite_loss_check_interval", 0) or 0
                )
                if nonfinite_interval > 0 and batch_idx % nonfinite_interval == 0:
                    local_finite = torch.isfinite(loss.detach()).to(
                        device=device,
                        dtype=torch.int32,
                    )
                    global_finite = local_finite.clone()
                    if dist.is_initialized():
                        dist.all_reduce(global_finite, op=dist.ReduceOp.MIN)
                    if int(global_finite.item()) == 0:
                        nonfinite_loss_streak += 1
                        loss_components = {
                            "loc": float(loss_loc.detach().float().item()),
                            "heat": float(loss_heatmap.detach().float().item()),
                            "star": float(loss_star.detach().float().item()),
                            "cons": float(loss_consistency.detach().float().item()),
                            "vis": float(loss_visibility.detach().float().item()),
                            "aux": float(loss_aux.detach().float().item()),
                        }
                        message = (
                            f"non-finite training loss at epoch {int(epoch)} "
                            f"batch {int(batch_idx)} (streak "
                            f"{nonfinite_loss_streak}/{MAX_NONFINITE_LOSS_STREAK}): "
                            f"local_finite={bool(int(local_finite.item()))} "
                            f"loss={float(loss.detach().float().item())} "
                            f"components={loss_components}"
                        )
                        if is_rank_zero():
                            log_event("train", message, level=Verbosity.QUIET)
                        if (
                            not scaler.is_enabled()
                            or nonfinite_loss_streak >= MAX_NONFINITE_LOSS_STREAK
                        ):
                            raise FloatingPointError(message)
                    else:
                        nonfinite_loss_streak = 0

                backward_start_time = start_timing(
                    device=device, synchronize=args.synchronize_runtime_timing
                )
                scaler.scale(loss).backward()
                if is_rank_zero():
                    accumulate_timing(
                        epoch_timing,
                        "backward_seconds",
                        backward_start_time,
                        device=device,
                        synchronize=args.synchronize_runtime_timing,
                    )

                optimizer_step_start_time = start_timing(
                    device=device, synchronize=args.synchronize_runtime_timing
                )
                scaler.unscale_(optimizer)

                grad_clip_start_time = start_timing(
                    device=device, synchronize=args.synchronize_runtime_timing
                )

                torch.nn.utils.clip_grad_norm_(
                    net.parameters(),
                    max_norm=1.0,
                )

                if is_rank_zero():
                    accumulate_timing(
                        epoch_timing,
                        "grad_clip_seconds",
                        grad_clip_start_time,
                        device=device,
                        synchronize=args.synchronize_runtime_timing,
                    )

                scaler.step(optimizer)
                if is_rank_zero():
                    accumulate_timing(
                        epoch_timing,
                        "optimizer_step_seconds",
                        optimizer_step_start_time,
                        device=device,
                        synchronize=args.synchronize_runtime_timing,
                    )

                scaler_update_start_time = start_timing(
                    device=device, synchronize=args.synchronize_runtime_timing
                )
                scaler.update()
                if is_rank_zero():
                    accumulate_timing(
                        epoch_timing,
                        "scaler_update_seconds",
                        scaler_update_start_time,
                        device=device,
                        synchronize=args.synchronize_runtime_timing,
                    )

                #                 loss.backward()
                #                 optimizer.step()

                ema.update_parameters(net.module)
                n += data.shape[0]
                if is_rank_zero() and train_progress is not None:
                    update_training_progress(
                        train_progress,
                        completed=batch_idx + 1,
                        total=total_train_steps,
                        count_completed=n,
                        count_total=len(train_dataset),
                    )
                if (
                    args.log_every > 0
                    and batch_idx % args.log_every == 0
                    and is_rank_zero()
                ):
                    loss_components = {
                        "loc": loss_loc.item(),
                        "heat": loss_heatmap.item(),
                        "star": loss_star.item(),
                        "cons": loss_consistency.item(),
                        "vis": loss_visibility.item(),
                        "aux": loss_aux.item(),
                    }
                    if train_progress is not None:
                        update_training_progress(
                            train_progress,
                            completed=batch_idx + 1,
                            total=total_train_steps,
                            count_completed=n,
                            count_total=len(train_dataset),
                            loss=loss.item(),
                            components=loss_components,
                        )
                    else:
                        train_line = (
                            f"e{int(epoch):03d} "
                            f"{int(batch_idx):06d}/{int(total_train_steps):06d} | "
                            f"{fmt_progress(batch_idx + 1, total_train_steps)} | "
                            f"loss {fmt_num(loss.item(), 3)} | "
                            f"{fmt_mapping(loss_components, precision=3, keys=_LOSS_COMPONENT_ORDER, omit_zero=True)}"
                        )
                        log_event(
                            "train",
                            train_line,
                            level=Verbosity.INFO,
                            loss=float(loss.item()),
                            components=loss_components,
                        )
                    if loss_details is not None and is_verbose():
                        _log_schema_head_details(loss_details)
                batch_fetch_start_time = time.time()

            stop_training_progress(train_progress)

            if hasattr(train_sampler, "last_epoch_diagnostics"):
                sampler_diagnostics = _aggregate_sampler_diagnostics(
                    train_sampler.last_epoch_diagnostics
                )
                if is_rank_zero():
                    log_event(
                        "sampler",
                        _sampler_summary_line(epoch, sampler_diagnostics),
                        level=Verbosity.VERBOSE,
                    )
                    log_event(
                        "sampler",
                        f"  full {sampler_diagnostics}",
                        level=Verbosity.DEBUG,
                    )
                    append_runtime_metrics(
                        args,
                        {
                            "event": "domain_balanced_sampler_epoch",
                            "epoch": int(epoch),
                            **sampler_diagnostics,
                        },
                    )

            if (
                args.save_legacy_epoch_state_dict
                and is_rank_zero()
                and args.save_n_epoch > 0
                and (epoch + 1) % args.save_n_epoch == 0
            ):
                if not os.path.exists(args.ckpt_folder):
                    os.mkdir(args.ckpt_folder)
                time_call(
                    epoch_timing,
                    "checkpoint_seconds",
                    torch.save,
                    net.module.state_dict(),
                    os.path.join(args.ckpt_folder, ("epoch_%d") % (epoch,)),
                )

            scheduler.step()

            global_train_samples = int(n)
            if dist.is_initialized():
                sample_count = torch.tensor(
                    [global_train_samples], device=device, dtype=torch.long
                )
                dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
                global_train_samples = int(sample_count.item())

            # All ranks must participate so full checkpoints can resume each rank's
            # own RNG state instead of replaying rank 0 RNG everywhere.
            _set_checkpoint_rng_state_by_rank(_collect_rng_state_by_rank())

            final_epoch = int(args.epoch) - 1
            eval_schedule = build_eval_schedule(
                args,
                epoch,
                final_epoch,
                limited_eval=eval_dataset is not test_dataset,
                has_ema=ema is not None,
            )
            should_eval_model = eval_schedule.should_eval_model
            eval_loader = (
                full_test_dataloader if eval_schedule.run_full_eval else test_dataloader
            )
            eval_scope = eval_schedule.eval_scope
            is_full_eval = eval_schedule.is_full_eval
            should_build_eval_records = eval_schedule.should_build_records
            model_report = None
            ema_report = None
            best_weights_state_dict = None

            if is_rank_zero():
                duration = time.time() - epoch_start_time
                samples_per_second = float(global_train_samples) / max(duration, 1e-9)
                peak_memory_mb = cuda_peak_memory_mb(device)
                current_lr = float(scheduler.get_last_lr()[0])
                epoch_line = (
                    f"{epoch:>3} done | {fmt_duration(duration)} | "
                    f"{fmt_count(global_train_samples)} samples | "
                    f"{fmt_num(samples_per_second, 1)} samples/s | "
                    f"lr {current_lr:.2e}"
                )
                if peak_memory_mb is not None:
                    epoch_line += f" | peak {fmt_num(peak_memory_mb, 1)} MB"
                log_event("epoch", epoch_line, level=Verbosity.QUIET)
                append_runtime_metrics(
                    args,
                    {
                        "epoch": int(epoch),
                        "duration_seconds": round(duration, 6),
                        "train_samples": int(global_train_samples),
                        "rank0_train_samples": int(n),
                        "samples_per_second": samples_per_second,
                        "peak_cuda_memory_mb": peak_memory_mb,
                        "lr": current_lr,
                    },
                )

                if eval_schedule.forced_final_full_eval:
                    log_event(
                        "eval",
                        "running full final eval so best.weights.pt and "
                        "best_checkpoint.pt are selected from the full "
                        "validation set",
                        level=Verbosity.VERBOSE,
                    )
                if should_eval_model and not should_build_eval_records:
                    log_event(
                        "eval",
                        f"fast overall-only eval at epoch {epoch}; slice "
                        f"reports every {args.eval_slice_reports_every} evaluated "
                        f"epoch(s)",
                        level=Verbosity.VERBOSE,
                    )

            if should_eval_model:
                eval_start_time = start_timing(
                    device=device, synchronize=args.synchronize_runtime_timing
                )
                with torch.inference_mode():
                    model_report = evaluate_landmark_model(
                        net.module,
                        eval_loader,
                        device,
                        include_records=bool(
                            args.eval_records_jsonl or args.eval_records_csv
                        ),
                        non_blocking=args.pin_memory,
                        build_records=should_build_eval_records,
                        show_progress=args.eval_progress and is_rank_zero(),
                        distributed=distributed_is_active(),
                    )
                eval_seconds = elapsed_timing(
                    eval_start_time,
                    device=device,
                    synchronize=args.synchronize_runtime_timing,
                )
                if is_rank_zero():
                    epoch_timing["eval_seconds"] = (
                        float(epoch_timing.get("eval_seconds", 0.0)) + eval_seconds
                    )
                    nme = model_report["overall"]["nme"]
                    if is_full_eval and nme is not None and best_nme > nme:
                        best_nme = nme
                        best_record.append((epoch, best_nme * 100))
                        best_weights_state_dict = net.module.state_dict()
                    print_eval_summary(f"test {eval_scope}", model_report)
                    log_event(
                        "eval",
                        f"best NME {fmt_num(best_nme * 100)}%",
                        level=Verbosity.QUIET,
                    )
                    append_runtime_metrics(
                        args,
                        {
                            "epoch": int(epoch),
                            "eval_scope": eval_scope,
                            "eval_seconds": round(eval_seconds, 6),
                            "eval_samples": int(
                                model_report["overall"].get("sample_count") or 0
                            ),
                        },
                    )
            else:
                if is_rank_zero():
                    log_event(
                        "eval",
                        f"skipping model eval at epoch {epoch}; "
                        f"--eval-every={args.eval_every}",
                        level=Verbosity.INFO,
                    )

            should_eval_ema = eval_schedule.should_eval_ema
            if should_eval_ema:
                ema_eval_start_time = start_timing(
                    device=device, synchronize=args.synchronize_runtime_timing
                )
                with torch.inference_mode():
                    ema_report = evaluate_landmark_model(
                        ema,
                        eval_loader,
                        device,
                        non_blocking=args.pin_memory,
                        build_records=should_build_eval_records,
                        show_progress=args.eval_progress and is_rank_zero(),
                        distributed=distributed_is_active(),
                    )
                if is_rank_zero():
                    accumulate_timing(
                        epoch_timing,
                        "ema_eval_seconds",
                        ema_eval_start_time,
                        device=device,
                        synchronize=args.synchronize_runtime_timing,
                    )
                    nme = ema_report["overall"]["nme"]
                    if is_full_eval and nme is not None and best_nme > nme:
                        best_nme = nme
                        best_record.append((epoch, "ema", best_nme * 100))
                        best_weights_state_dict = ema.model.state_dict()
                    print_eval_summary(f"test ema {eval_scope}", ema_report)
                    best_history = " | ".join(
                        (
                            f"epoch {int(entry[0])} ({entry[1]}) {fmt_num(entry[-1], 2)}%"
                            if len(entry) == 3
                            else f"epoch {int(entry[0])} {fmt_num(entry[-1], 2)}%"
                        )
                        for entry in best_record
                    )
                    log_event(
                        "eval",
                        f"best history: {best_history or '-'}",
                        level=Verbosity.VERBOSE,
                    )
            elif ema is not None and should_eval_model:
                if is_rank_zero():
                    reason = (
                        eval_schedule.ema_skip_reason
                        or f"--eval-ema-every={args.eval_ema_every}"
                    )
                    log_event(
                        "eval",
                        f"skipping EMA eval at epoch {epoch}; {reason}",
                        level=Verbosity.INFO,
                    )

            if is_rank_zero():
                if best_weights_state_dict is not None:
                    time_call(
                        epoch_timing,
                        "checkpoint_seconds",
                        _save_best_weights,
                        best_weights_state_dict,
                        args.ckpt_folder,
                    )
                    time_call(
                        epoch_timing,
                        "checkpoint_seconds",
                        _save_training_checkpoint,
                        Path(args.ckpt_folder) / "best_checkpoint.pt",
                        net,
                        optimizer,
                        scheduler,
                        scaler,
                        ema,
                        epoch,
                        best_nme,
                        best_record,
                        args,
                    )
                if args.save_n_epoch > 0 and (epoch + 1) % args.save_n_epoch == 0:
                    time_call(
                        epoch_timing,
                        "checkpoint_seconds",
                        _save_training_checkpoint,
                        Path(args.ckpt_folder) / f"checkpoint_epoch_{epoch:04d}.pt",
                        net,
                        optimizer,
                        scheduler,
                        scaler,
                        ema,
                        epoch,
                        best_nme,
                        best_record,
                        args,
                    )
                if model_report is not None:
                    records = records_from_report(model_report)
                    compact_model_report = {
                        key: value
                        for key, value in model_report.items()
                        if key != "records"
                    }
                    eval_payload = {
                        "epoch": epoch,
                        "eval_mode": args.eval_mode,
                        "eval_scope": eval_scope,
                        "heldout_datasets": list(args.heldout_dataset),
                        "model": compact_model_report,
                    }
                    if ema_report is not None:
                        eval_payload["ema"] = ema_report
                    write_eval_json(eval_report_json_path(args), eval_payload)
                    if args.eval_report_csv:
                        write_eval_csv(args.eval_report_csv, eval_payload)
                    if args.eval_records_jsonl:
                        write_eval_records_jsonl(args.eval_records_jsonl, records)
                    if args.eval_records_csv:
                        write_eval_records_csv(args.eval_records_csv, records)
                # Save last checkpoint after eval so best_nme and best_record are current.
                if args.save_last_checkpoint:
                    time_call(
                        epoch_timing,
                        "checkpoint_seconds",
                        _save_training_checkpoint,
                        Path(args.ckpt_folder) / "last_checkpoint.pt",
                        net,
                        optimizer,
                        scheduler,
                        scaler,
                        ema,
                        epoch,
                        best_nme,
                        best_record,
                        args,
                        weights_path=Path(args.ckpt_folder)
                        / "last_checkpoint.weights.pt",
                    )

            _distributed_barrier()
            if is_rank_zero():
                final_epoch_timing = finalize_epoch_timing(
                    epoch_timing,
                    epoch_wall_seconds=time.time() - epoch_start_time,
                )
                append_runtime_metrics(
                    args,
                    {
                        "event": "epoch_timing",
                        "epoch": int(epoch),
                        "timing": {
                            key: round(float(value), 6)
                            for key, value in sorted(final_epoch_timing.items())
                        },
                    },
                )

                if epoch + 1 >= args.epoch:
                    _write_training_complete_sentinel(
                        args,
                        epoch,
                        best_nme,
                        best_record,
                        global_train_samples,
                    )


if __name__ == "__main__":
    main()
