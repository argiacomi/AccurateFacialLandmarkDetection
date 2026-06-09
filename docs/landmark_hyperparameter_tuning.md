# Landmark hyperparameter tuning

Issue #7 adds a reproducible tuning orchestrator:

```bash
python tools/tune_training_hyperparameters.py \
  --output-dir runs/hparam_tuning \
  --dry-run
```

The script writes every generated config, command line, metric file, ranked trial list, finalist summary, and final recommendation under the output directory. Re-running with the same output directory resumes from existing `result.json` artifacts and the persisted Optuna study.

## Stages

The staged plan is:

1. **Baseline** using fixed default loss weights and learning rate.
2. **Manual STARLoss_v2 bracket** with fixed consistency, auxiliary, and LR values.
3. **Optuna narrow loss-weight search** over:
   - `star_loss_weight`
   - `schema_consistency_weight`
   - `auxiliary_loss_weight`
4. **Multi-seed loss-weight finalist reruns** for the top configs.
5. **Learning-rate sweep** around `1e-4` with selected loss weights frozen.
6. **Multi-seed LR finalist reruns**.
7. **Final recommendation** in `best_training_hyperparameters.json`.

## Optuna study

`requirements.txt` includes Optuna. By default, the loss-weight stage creates or resumes a SQLite-backed study at:

```text
sqlite:///<output-dir>/optuna_study.db
```

The script uses Optuna's `create_study(..., load_if_exists=True)`, `study.ask()`, and `study.tell()` flow so trial commands can be launched by the orchestrator while scores are persisted back to Optuna after metrics are available.

Useful options:

```text
--optuna-trials 40
--optuna-study-name landmark_loss_weight_search
--optuna-storage sqlite:////shared/path/landmark_tuning.db
--optuna-workers 4
--optuna-min-pruning-epoch 5
--optuna-pruner-startup-trials 5
--require-optuna
--disable-optuna
```

For parallel workers, launch multiple script processes with the same `--output-dir` or shared `--optuna-storage`; Optuna coordinates trial assignment through the shared storage.

`--require-optuna` fails fast if Optuna cannot be imported. `--disable-optuna` is only for minimal environments and uses deterministic fallback sampling instead of a real study.

## Dry-run planning

Dry-run mode prints and stores commands without launching training:

```bash
python tools/tune_training_hyperparameters.py \
  --output-dir runs/hparam_tuning \
  --dry-run \
  --train-command "python TrainHeatmapStageFP16.py" \
  --extra-train-args "--data_name FS68Manifest --manifest data/train.json --test_manifest data/val.json"
```

Each run receives flags like:

```text
--star-loss-weight ...
--schema-consistency-weight ...
--auxiliary-loss-weight ...
--lr ...
--locw ...
--hw ...
--seed ...
--ckpt_folder <run>/checkpoints
--eval-report-json <run>/metrics.json
--runtime-metrics-jsonl <run>/runtime_metrics.jsonl
```

## Execute real training

Use `--execute` to launch the generated commands:

```bash
python tools/tune_training_hyperparameters.py \
  --output-dir runs/hparam_tuning \
  --execute \
  --train-command "python TrainHeatmapStageFP16.py" \
  --extra-train-args "--data_name FS68Manifest --manifest data/train.json --test_manifest data/val.json --epoch 20"
```

Training/evaluation should write the evaluation JSON to the path passed through `--eval-report-json`. The tuner accepts already-flat metric JSON, and also normalizes the trainer's nested eval report under `model.overall` and slice groups such as `model.by_hard_negative_bucket`.

## Objective

The score is lower-is-better:

```text
score =
  heldout_68_nme
+ hard_slice_weight * profile_nme
+ hard_slice_weight * occlusion_nme
+ hard_slice_weight * profile_occlusion_nme
+ optional blur/low-quality terms
+ regression penalties versus baseline
```

Missing hard slices are reported in each result's objective diagnostics instead of silently disappearing.

## Mock metrics smoke test

For planner and CI smoke tests, `--mock-metrics` writes deterministic synthetic metrics instead of launching training while still exercising Optuna trial creation when Optuna is installed:

```bash
python tools/tune_training_hyperparameters.py \
  --output-dir /tmp/landmark_tuning_smoke \
  --dry-run \
  --mock-metrics \
  --optuna-trials 4 \
  --star-bracket 0,0.005,0.01 \
  --lr-sweep 0.00005,0.0001,0.0002
```

## Key outputs

- `baseline_config.json`
- `baseline_result.json`
- `optuna_study.db`
- `optuna_trial_plan.json`
- `optuna_study.json`
- `ranked_loss_candidates.json`
- `loss_finalist_summary.json`
- `ranked_lr_candidates.json`
- `lr_finalist_summary.json`
- `best_training_hyperparameters.json`
- `runs/<run-id>/config.json`
- `runs/<run-id>/command.txt`
- `runs/<run-id>/metrics.json`
- `runs/<run-id>/runtime_metrics.jsonl`
- `runs/<run-id>/result.json`

The final JSON contains the selected values for:

- `star_loss_weight`
- `schema_consistency_weight`
- `auxiliary_loss_weight`
- `locw`
- `hw`
- `lr`

It also includes the exact training flags to reuse for the selected configuration.
