"""CD-ViT heatmap training implementation.

This module backs the legacy TrainHeatmapStageFP16.py entrypoint.  Keep
long-lived training logic here and keep TrainHeatmapStageFP16.py as a thin
compatibility wrapper for existing commands, pipeline invocations, and
checkpoint metadata.
"""

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
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.optim.lr_scheduler import StepLR

from lib.landmarks.core.manifest_aliases import (
    CANONICAL_MANIFEST_DATA_NAME,
    LEGACY_MANIFEST_DATA_NAME,
    is_schema_aware_manifest_dataset,
)
from lib.landmarks.evaluation.split_safe import (
    write_eval_csv,
    write_eval_json,
    write_eval_records_csv,
    write_eval_records_jsonl,
)
from lib.landmarks.training.checkpoint_compat import (
    checkpoint_compat_errors_from_args as _checkpoint_compat_errors,
)
from lib.landmarks.training.checkpointing import (
    _collect_rng_state_by_rank,
    _restore_training_checkpoint,
    _save_training_checkpoint,
    _set_checkpoint_rng_state_by_rank,
    _torch_load_training_checkpoint,
    _write_training_complete_sentinel,
)
from lib.landmarks.training.cli import build_heatmap_stage_arg_parser
from lib.landmarks.training.config import (
    DatasetBuildConfig,
    EvalConfig,
    TrainingRuntimeConfig,
    config_dict,
)
from lib.landmarks.training.data import (
    AUXILIARY_CLASS_NAMES,
    batch_mix,
    build_dataset,
    landmark_count_for_dataset,
    schema_aware_collate,
    unpack_train_batch,
)
from lib.landmarks.training.ddp import (
    distributed_rank,
    distributed_world_size,
    is_rank_zero,
    setup_distributed_from_env,
)
from lib.landmarks.training.ema import EMA
from lib.landmarks.training.eval_schedule import build_eval_schedule
from lib.landmarks.training.evaluator import (
    eval_collate,
    eval_report_json_path,
    evaluate_landmark_model,
    print_eval_summary,
    records_from_report,
)
from lib.landmarks.training.loaders import build_training_loaders
from lib.landmarks.training.losses import (
    heatmap_batch_weight,
    schema_head_loss,
    weighted_smooth_l1,
)
from lib.landmarks.training.model_factory import build_cdvit_model
from lib.landmarks.training.profiling import (
    accumulate_timing,
    append_runtime_metrics,
    cuda_peak_memory_mb,
    elapsed_timing,
    empty_epoch_timing,
    finalize_epoch_timing,
    start_timing,
    time_call,
)
from lib.landmarks.training.runtime import (
    dataloader_kwargs,
    maybe_limit_eval_dataset,
    normalize_runtime_args,
    set_dataset_runtime_epoch,
    should_run_interval,
)
from lib.landmarks.training.seed import setup_seed
from loss import AWingLoss
from loss import STARLoss_v2

# from torch.cuda.amp import autocast as autocast


LEGACY_FS68_DATASET_NAME = LEGACY_MANIFEST_DATA_NAME
MULTI_SCHEMA_MANIFEST_DATASET_NAME = CANONICAL_MANIFEST_DATA_NAME
FS68_DATASET_NAME = LEGACY_FS68_DATASET_NAME

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


def _save_best_weights(state_dict, ckpt_folder: str | os.PathLike[str]) -> None:
    """Write explicit best weights and the legacy best_model compatibility copy."""

    ckpt_path = Path(ckpt_folder)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, ckpt_path / "best.weights.pt")
    torch.save(state_dict, ckpt_path / "best_model")


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
        print(
            "resume checkpoint missing schema/auxiliary head keys initialized from current model: "
            f"{len(missing)} missing ({missing[:10]})",
            flush=True,
        )
    if unexpected:
        print(
            "resume checkpoint unexpected keys ignored: "
            f"{len(unexpected)} unexpected ({unexpected[:10]})",
            flush=True,
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
    setup_seed(args.seed, deterministic=args.deterministic)
    lmk_num = landmark_count_for_dataset(args)
    device = setup_distributed_from_env(args)
    training_runtime_config = TrainingRuntimeConfig.from_args(args)
    eval_config = EvalConfig.from_args(args)
    dataset_build_config = DatasetBuildConfig.from_args(args)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
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
        print("----------------------len(train_dataset)", len(train_dataset))
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
        ).cuda()
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
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[args.local_rank],
            find_unused_parameters=args.find_unused_parameters or schema_aware_training,
        )

        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, args.sched_step, gamma=0.5)

        best_nme = 99999
        weights = [1 / math.pow(args.mul, i) for i in range(args.nstack)]
        weights.reverse()
        best_record = []

        # heatmap_loss_func = HeatMapLoss2
        heatmap_loss_func = AWingLoss()
        vertex_loss_func = STARLoss_v2() if args.star_loss_weight > 0 else None
        ema = EMA(net.module, 0.99, 100, 10) if is_rank_zero() else None
        scaler = torch.amp.GradScaler("cuda")
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
                print(f"resumed training checkpoint from epoch {start_epoch}")
        if start_epoch >= args.epoch:
            if is_rank_zero():
                print(
                    f"resume checkpoint next_epoch={start_epoch} is >= requested epoch={args.epoch}; "
                    "training is already complete for this epoch target",
                    flush=True,
                )
            if dist.is_initialized():
                barrier_start = time.time()
                dist.barrier()
                wait_seconds = time.time() - barrier_start
                local_wait = torch.tensor(
                    [wait_seconds], device=device, dtype=torch.float64
                )
                gathered_waits = [
                    torch.zeros_like(local_wait)
                    for _ in range(distributed_world_size())
                ]
                dist.all_gather(gathered_waits, local_wait)
                if is_rank_zero():
                    waits = [float(value.item()) for value in gathered_waits]
                    append_runtime_metrics(
                        args,
                        {
                            "event": "distributed_eval_wait",
                            "epoch": int(start_epoch),
                            "distributed_eval_wait_seconds": round(max(waits), 6),
                            "distributed_eval_wait_seconds_by_rank": {
                                str(rank): round(wait, 6)
                                for rank, wait in enumerate(waits)
                            },
                        },
                    )
            return
        for epoch in range(start_epoch, args.epoch):
            args.current_epoch = int(epoch)
            n = 0
            net.train()
            if is_rank_zero():
                ema.train()
            if is_rank_zero():
                epoch_start_time = time.time()
                epoch_timing = empty_epoch_timing()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)
            train_sampler.set_epoch(epoch)
            set_dataset_runtime_epoch(train_dataset, epoch, args)
            if args.persistent_workers and epoch == start_epoch and is_rank_zero():
                print(
                    "note: --persistent-workers seeds DataLoader workers once for throughput; "
                    "use --no-persistent-workers for epoch-reseeded worker RNG",
                    flush=True,
                )
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
                with torch.autocast(device_type="cuda", dtype=torch.float16):
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

                if dist.get_rank() == 0:
                    ema.update_parameters(net.module)
                n += data.shape[0]
                if (
                    args.log_every > 0
                    and batch_idx % args.log_every == 0
                    and is_rank_zero()
                ):
                    mix = batch_mix(batch)
                    mix_text = f" mix: {mix}" if mix else ""
                    print(
                        f"train epoch {epoch} batch_idx {batch_idx} rank {distributed_rank()}  {n}/{len(train_dataset)} "
                        f"loss: {loss.item()} loss_loc: {loss_loc.item()} loss_heatmap: {loss_heatmap.item()} "
                        f"loss_consistency: {loss_consistency.item()} loss_star: {loss_star.item()} "
                        f"loss_visibility: {loss_visibility.item()} loss_aux: {loss_aux.item()}{mix_text}"
                    )
                    if loss_details is not None:
                        head_counts = loss_details.get("head_sample_counts", {})
                        head_losses = {
                            name: float(value.item())
                            for name, value in loss_details.get(
                                "head_loss_contributions", {}
                            ).items()
                        }
                        print(
                            f"schema head loss details counts={head_counts} contributions={head_losses} "
                            f"aux_valid={loss_details.get('auxiliary_valid_counts', {})} "
                            f"aux_accuracy={loss_details.get('auxiliary_accuracy', {})} "
                            f"visibility_valid={loss_details.get('visibility_valid_counts', {})} "
                            f"visibility_weight={float(loss_details.get('visibility_loss_weight', torch.tensor(0.0)).item())}",
                            flush=True,
                        )
                batch_fetch_start_time = time.time()

            if hasattr(train_sampler, "last_epoch_diagnostics"):
                sampler_diagnostics = _aggregate_sampler_diagnostics(
                    train_sampler.last_epoch_diagnostics
                )
                if is_rank_zero():
                    print(
                        f"domain-balanced sampler epoch {epoch}: {sampler_diagnostics}",
                        flush=True,
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

            if is_rank_zero():
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
                duration = time.time() - epoch_start_time
                samples_per_second = float(global_train_samples) / max(duration, 1e-9)
                peak_memory_mb = cuda_peak_memory_mb(device)
                print(
                    f"#epoch runtime epoch={epoch} duration={duration:.3f}s "
                    f"samples_per_second={samples_per_second:.3f} "
                    f"peak_cuda_memory_mb={peak_memory_mb}"
                )
                append_runtime_metrics(
                    args,
                    {
                        "epoch": int(epoch),
                        "duration_seconds": round(duration, 6),
                        "train_samples": int(global_train_samples),
                        "rank0_train_samples": int(n),
                        "samples_per_second": samples_per_second,
                        "peak_cuda_memory_mb": peak_memory_mb,
                        "lr": float(scheduler.get_last_lr()[0]),
                    },
                )

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
                    full_test_dataloader
                    if eval_schedule.run_full_eval
                    else test_dataloader
                )
                eval_scope = eval_schedule.eval_scope
                is_full_eval = eval_schedule.is_full_eval
                should_build_eval_records = eval_schedule.should_build_records
                if eval_schedule.forced_final_full_eval:
                    print(
                        "running full final eval so best.weights.pt and best_checkpoint.pt are selected "
                        "from the full validation set"
                    )
                model_report = None
                ema_report = None
                if should_eval_model and not should_build_eval_records:
                    print(
                        f"running fast overall-only eval at epoch {epoch}; "
                        f"slice reports every {args.eval_slice_reports_every} evaluated epoch(s)"
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
                            show_progress=args.eval_progress,
                        )
                    eval_seconds = elapsed_timing(
                        eval_start_time,
                        device=device,
                        synchronize=args.synchronize_runtime_timing,
                    )
                    epoch_timing["eval_seconds"] = (
                        float(epoch_timing.get("eval_seconds", 0.0)) + eval_seconds
                    )
                    nme = model_report["overall"]["nme"]
                    if is_full_eval and nme is not None and best_nme > nme:
                        best_nme = nme
                        best_record.append((epoch, best_nme * 100))
                        time_call(
                            epoch_timing,
                            "checkpoint_seconds",
                            _save_best_weights,
                            net.module.state_dict(),
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
                    print_eval_summary(f"test {eval_scope}", model_report)
                    print("BEST NME %: {}".format(best_nme * 100))
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
                    print(
                        f"skipping model eval at epoch {epoch}; --eval-every={args.eval_every}"
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
                            show_progress=args.eval_progress,
                        )
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
                        time_call(
                            epoch_timing,
                            "checkpoint_seconds",
                            _save_best_weights,
                            ema.model.state_dict(),
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
                    print_eval_summary(f"test ema {eval_scope}", ema_report)
                    print(best_record)
                elif ema is not None and should_eval_model:
                    reason = (
                        eval_schedule.ema_skip_reason
                        or f"--eval-ema-every={args.eval_ema_every}"
                    )
                    print(f"skipping EMA eval at epoch {epoch}; {reason}")

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
                    )

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

            if dist.is_initialized():
                dist.barrier()


if __name__ == "__main__":
    main()
