"""Evaluation scheduling helpers for heatmap-stage training."""

from __future__ import annotations

from dataclasses import dataclass
import typing as T

from lib.landmarks.training.runtime import should_run_interval


@dataclass(frozen=True)
class EvalSchedule:
    should_eval_model: bool
    run_full_eval: bool
    forced_final_full_eval: bool
    eval_scope: str
    is_full_eval: bool
    should_build_records: bool
    should_eval_ema: bool
    ema_skip_reason: str = ""

    @property
    def fast_overall_only(self) -> bool:
        return self.should_eval_model and not self.should_build_records


def _ema_due_for_scope(
    scope_policy: str, *, is_full_eval: bool, final_epoch: bool
) -> tuple[bool, str]:
    if scope_policy in {"", "same"}:
        return True, ""
    if scope_policy == "full-only":
        return is_full_eval, "eval scope is sampled and --eval-ema-scope=full-only"
    if scope_policy == "final-only":
        return final_epoch, "epoch is not final and --eval-ema-scope=final-only"
    if scope_policy == "off":
        return False, "--eval-ema-scope=off"
    raise ValueError(
        "--eval-ema-scope must be one of: same, full-only, final-only, off"
    )


def build_eval_schedule(
    args: T.Any,
    epoch: int,
    final_epoch: int,
    *,
    limited_eval: bool,
    has_ema: bool,
) -> EvalSchedule:
    """Return the eval work that should run for the epoch.

    The key overhead reduction is that EMA evaluation can be scoped to full evals
    or the final epoch. Pipeline runs use full-only by default so sampled evals do
    not run both the model and EMA over the same sampled validation subset.
    """

    should_eval_model = should_run_interval(args.eval_every, epoch, final_epoch)
    run_full_eval = should_run_interval(args.full_eval_every, epoch, final_epoch)
    forced_final_full_eval = False
    if (
        limited_eval
        and should_eval_model
        and epoch >= final_epoch
        and not run_full_eval
    ):
        run_full_eval = True
        forced_final_full_eval = True

    eval_scope = "full" if (run_full_eval or not limited_eval) else "sampled"
    is_full_eval = eval_scope == "full"
    should_build_records = bool(
        args.eval_records_jsonl
        or args.eval_records_csv
        or args.eval_report_csv
        or should_run_interval(args.eval_slice_reports_every, epoch, final_epoch)
    )

    ema_period_due = bool(
        has_ema
        and should_eval_model
        and should_run_interval(args.eval_ema_every, epoch, final_epoch)
    )
    scope_policy = getattr(args, "eval_ema_scope", "same")
    scope_due, scope_skip_reason = _ema_due_for_scope(
        scope_policy,
        is_full_eval=is_full_eval,
        final_epoch=epoch >= final_epoch,
    )
    should_eval_ema = bool(ema_period_due and scope_due)
    ema_skip_reason = ""
    if has_ema and should_eval_model and not should_eval_ema:
        if not ema_period_due:
            ema_skip_reason = f"--eval-ema-every={args.eval_ema_every}"
        else:
            ema_skip_reason = scope_skip_reason

    return EvalSchedule(
        should_eval_model=should_eval_model,
        run_full_eval=run_full_eval,
        forced_final_full_eval=forced_final_full_eval,
        eval_scope=eval_scope,
        is_full_eval=is_full_eval,
        should_build_records=should_build_records,
        should_eval_ema=should_eval_ema,
        ema_skip_reason=ema_skip_reason,
    )


__all__ = ["EvalSchedule", "build_eval_schedule"]
