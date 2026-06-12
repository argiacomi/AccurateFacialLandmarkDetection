"""CLI construction for CD-ViT heatmap-stage training."""

from __future__ import annotations

import argparse

from lib.evaluation.split_safe import EVAL_MODES, SPLIT_POLICIES


def build_heatmap_stage_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, default="")
    parser.add_argument("--ckpt_folder", type=str, default="checkpoint")
    parser.add_argument("--batch_size", type=int, default="16")
    parser.add_argument("--num_workers", type=int, default="12")
    parser.add_argument("--epoch", type=int, default="500")
    parser.add_argument("--lr", type=float, default="0.0001")
    parser.add_argument("--local_rank", type=int, help="local rank, will passed by ddp")
    parser.add_argument("--local-rank", type=int, help="local rank, will passed by ddp")
    parser.add_argument("--sched_step", type=int, default="200")
    parser.add_argument("--save_n_epoch", type=int, default="100")
    parser.add_argument(
        "--preload",
        type=int,
        default="0",
        help="0 streams samples through DataLoader workers; 1 preloads the dataset in memory.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin DataLoader host memory so CUDA transfers can use non_blocking=True.",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep DataLoader workers alive for throughput. Worker RNG is seeded once; use --no-persistent-workers for epoch-reseeded workers or --restore-rng replay.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader batches prefetched per worker. Ignored when num_workers == 0.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--full-eval-every", type=int, default=0)
    parser.add_argument(
        "--eval-ema-every",
        "--eval-on-ema-every",
        dest="eval_ema_every",
        type=int,
        default=1,
    )
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument(
        "--eval-slice-reports-every",
        type=int,
        default=1,
        help=(
            "Build per-sample slice reports every N evaluated epochs. "
            "Default 1 preserves direct trainer behavior; pipeline runs "
            "override this to a larger value for throughput."
        ),
    )
    parser.add_argument(
        "--eval-ema-scope",
        choices=("same", "full-only", "final-only", "off"),
        default="same",
        help=(
            "Controls when EMA evaluation runs after --eval-ema-every is due. "
            "same preserves legacy behavior; full-only skips EMA on sampled evals; "
            "final-only runs EMA only on the final epoch; off disables EMA eval."
        ),
    )
    parser.add_argument(
        "--eval-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Show an interactive Rich progress bar during evaluation. "
            "Disabled by default to keep logs compact."
        ),
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--train-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Show a Rich live training progress bar when running interactively. "
            "Non-TTY logs and --log-format json fall back to periodic train lines."
        ),
    )
    parser.add_argument(
        "--save-last-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write <ckpt_folder>/last_checkpoint.pt after each epoch.",
    )
    parser.add_argument(
        "--save-legacy-epoch-state-dict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write legacy epoch_N model state-dict files at --save_n_epoch intervals.",
    )
    parser.add_argument(
        "--runtime-metrics-jsonl",
        type=str,
        default="",
        help="Optional runtime metrics JSONL path. Defaults to <ckpt_folder>/runtime_metrics.jsonl.",
    )
    parser.add_argument(
        "--restore-rng",
        action="store_true",
        help="Restore RNG state from full checkpoints. For exact replay this forces --no-persistent-workers so workers are re-seeded per epoch.",
    )
    parser.add_argument(
        "--allow-incompatible-resume",
        action="store_true",
        help="Allow loading a full checkpoint even when manifest/config compatibility metadata differs.",
    )
    parser.add_argument(
        "--allow-missing-schema-heads",
        action="store_true",
        help="Allow intentionally resuming an older 68-only checkpoint into a schema-aware model by initializing missing schema/auxiliary heads.",
    )
    parser.add_argument("--hw", type=float, default="10")
    parser.add_argument("--locw", type=float, default="1")
    parser.add_argument("--nstack", type=int, default="8")
    parser.add_argument("--heatmap_size", type=int, default="32")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max_depth", type=int, default="256")
    parser.add_argument("--mul", type=float, default="1.2")
    parser.add_argument(
        "--lmk_num",
        type=int,
        default="68",
        help="fallback landmark count for schema-aware manifest aliases",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="schema-aware landmark manifest for train/test",
    )
    parser.add_argument(
        "--train_manifest", type=str, default="", help="schema-aware train manifest"
    )
    parser.add_argument(
        "--test_manifest", type=str, default="", help="schema-aware test manifest"
    )
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
    parser.add_argument(
        "--eval-report-json",
        type=str,
        default="",
        help="Evaluation JSON path. Defaults to <ckpt_folder>/eval_report.json",
    )
    parser.add_argument(
        "--eval-report-csv", type=str, default="", help="Optional evaluation CSV path"
    )
    parser.add_argument(
        "--eval-records-jsonl",
        type=str,
        default="",
        help="Optional per-sample evaluation records JSONL path",
    )
    parser.add_argument(
        "--eval-records-csv",
        type=str,
        default="",
        help="Optional per-sample evaluation records CSV path",
    )
    parser.add_argument(
        "--schema-aware-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For schema-aware manifest aliases, train schema-specific heads from mixed-schema manifests.",
    )
    parser.add_argument("--schema-consistency-weight", type=float, default=0.05)
    parser.add_argument(
        "--schema-head-loss-weighting",
        choices=("sample_count", "per_head"),
        default="sample_count",
        help="Weight mixed schema-head losses by supervised sample count or by one mean loss per active head.",
    )
    parser.add_argument(
        "--schema-head-loss-weights",
        default="",
        help="Optional comma-separated per-head multipliers, e.g. landmarks_98=1.0,profile39=0.5.",
    )
    parser.add_argument(
        "--star-loss-weight",
        type=float,
        default=0.0,
        help="Optional small STARLoss_v2 regularizer for active supervised schema heads; try 0.005, 0.01, 0.02, or 0.05 on hard-case slices.",
    )
    parser.add_argument(
        "--star-loss-check-finite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Guard against NaN/Inf in the STARLoss_v2 covariance before "
            "eigendecomposition. Off by default because the check forces a CUDA "
            "host-device sync every step; enable for debugging/CI/smoke runs."
        ),
    )
    parser.add_argument(
        "--star-loss-check-finite-interval",
        type=int,
        default=0,
        help=(
            "When --star-loss-check-finite is set, run the NaN/Inf guard only "
            "every N STAR forwards (e.g. 100) instead of every step, so long "
            "runs keep protection without paying the sync each step. 0 (default) "
            "checks every step."
        ),
    )
    parser.add_argument(
        "--nonfinite-loss-check-interval",
        type=int,
        default=0,
        help=(
            "Check total loss for NaN/Inf every N training batches. "
            "0 disables this extra synchronized guard; use a positive value "
            "for tuner/debug runs where fast failure is worth the sync."
        ),
    )
    parser.add_argument(
        "--domain-balanced-sampling",
        action="store_true",
        help=(
            "Use balanced per-rank batches under DDP; --batch_size remains the "
            "per-rank batch size and all ranks get the same number of steps."
        ),
    )
    parser.add_argument(
        "--bucket-targets",
        default="anchor=0.25,occlusion=0.25,profile=0.25,profile_occlusion=0.25",
        help="Comma-separated hard bucket target weights for domain-balanced sampling.",
    )
    parser.add_argument(
        "--dataset-targets", default="", help="Comma-separated dataset target weights."
    )
    parser.add_argument(
        "--schema-targets", default="", help="Comma-separated schema target weights."
    )
    parser.add_argument(
        "--auto-dataset-balancing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Infer uniform dataset targets from observed train samples when --dataset-targets is empty.",
    )
    parser.add_argument(
        "--auto-schema-balancing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Infer uniform schema/head targets from observed train samples when --schema-targets is empty.",
    )
    parser.add_argument(
        "--auxiliary-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable optional pose/quality auxiliary heads for schema-aware manifest training.",
    )
    parser.add_argument("--auxiliary-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--auxiliary-loss-weights",
        default="",
        help="Optional comma-separated per-task auxiliary loss weights, e.g. occlusion=1.0,blur_quality=0.5.",
    )
    parser.add_argument(
        "--auxiliary-loss-stage",
        choices=("all", "final"),
        default="final",
        help="Apply auxiliary and visibility losses on all stacks or final stack only.",
    )
    parser.add_argument(
        "--visibility-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable schema-aware per-point visibility heads.",
    )
    parser.add_argument("--visibility-loss-weight", type=float, default=0.0)
    parser.add_argument("--visibility-loss-initial-weight", type=float, default=0.0)
    parser.add_argument("--visibility-loss-start-epoch", type=int, default=1)
    parser.add_argument("--visibility-loss-ramp-epochs", type=int, default=0)
    parser.add_argument("--visibility-pseudo-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--visibility-detach-heatmaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Detach schema-head landmark heatmaps before landmark-conditioned "
            "visibility pooling. Default keeps noisy visibility gradients from "
            "updating landmark localization heads; pass --no-visibility-detach-heatmaps "
            "to ablate joint visibility/localization gradients."
        ),
    )
    parser.add_argument(
        "--synchronize-runtime-timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Synchronize CUDA around timed transfer/compute/eval sections for more accurate profiling. "
            "Pass --no-synchronize-runtime-timing for low-overhead CPU wall-clock timing."
        ),
    )
    parser.add_argument("--data_name", type=str, default="FS68Manifest")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help=(
            "Accelerator to train on. 'auto' prefers CUDA, then Apple Silicon "
            "(MPS), then CPU. CUDA enables NCCL DDP (under torchrun), fp16 AMP, "
            "and FlashAttention; MPS/CPU run single-process in fp32."
        ),
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compile the model with torch.compile for faster training. Adds "
            "warmup compilation cost on the first steps; variable batch "
            "composition (e.g. domain-balanced sampling) may trigger recompiles."
        ),
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help=(
            "torch.compile optimization mode; only used with --compile and the "
            "Inductor backend."
        ),
    )
    parser.add_argument(
        "--compile-backend",
        type=str,
        default="auto",
        choices=["auto", "inductor", "aot_eager", "eager"],
        help=(
            "torch.compile backend. 'auto' uses Inductor on CUDA/CPU and "
            "aot_eager on MPS (Inductor's MPS backend is experimental and can "
            "miscompile conv backward). Only used with --compile."
        ),
    )
    parser.add_argument("--seed", type=int, default="0")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable deterministic cuDNN behavior. Default favors training throughput.",
    )
    parser.add_argument(
        "--find_unused_parameters",
        action="store_true",
        help="Enable only if the model forward pass can skip trainable parameters",
    )
    parser.add_argument(
        "--log-format",
        type=str,
        default="human",
        choices=["human", "json"],
        help=(
            "Console output format. 'human' prints short tagged lines; 'json' "
            "emits one JSON object per event for CI/log parsing. The structured "
            "runtime_metrics.jsonl is written either way."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["quiet", "info", "verbose", "debug"],
        help=(
            "Console verbosity. 'quiet' shows only epoch/eval summaries and "
            "errors; 'info' shows the Rich train progress bar or compact periodic "
            "train lines; 'verbose' adds head diagnostics, sampler detail, and "
            "checkpoint writes; 'debug' adds full structures."
        ),
    )
    return parser


__all__ = ["build_heatmap_stage_arg_parser"]
