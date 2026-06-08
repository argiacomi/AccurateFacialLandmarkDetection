# Landmark hyperparameter tuning

Issue #7 adds a reproducible tuning orchestrator:

```bash
python tools/landmarks/tune_training_hyperparameters.py \
  --output-dir runs/hparam_tuning \
  --dry-run
```

The script writes every generated config, command line, metric file, ranked trial list, finalist summary, and final recommendation under the output directory. Re-running with the same output directory resumes from existing `result.json` artifacts.

## Stages

The staged plan is:

1. **Baseline** using fixed default loss weights and learning rate.
2. **Manual STARLoss_v2 bracket** with fixed consistency, auxiliary, and LR values.
3. **Narrow loss-weight search** over:
   - `star_loss_weight`
   - `schema_consistency_weight`
   - `auxiliary_loss_weight`
4. **Multi-seed loss-weight finalist reruns** for the top configs.
5. **Learning-rate sweep** around `1e-4` with selected loss weights frozen.
6. **Multi-seed LR finalist reruns**.
7. **Final recommendation** in `best_training_hyperparameters.json`.

## Dry-run planning

Dry-run mode prints and stores commands without launching training:

```bash
python tools/landmarks/tune_training_hyperparameters.py \
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
--runtime-metrics-path <run>/metrics.json
```

## Execute real training

Use `--execute` to launch the generated commands:

```bash
python tools/landmarks/tune_training_hyperparameters.py \
  --output-dir runs/hparam_tuning \
  --execute \
  --train-command "python TrainHeatmapStageFP16.py" \
  --extra-train-args "--data_name FS68Manifest --manifest data/train.json --test_manifest data/val.json --epoch 20"
```

Training/evaluation should write JSON metrics to the path passed through `--runtime-metrics-path`. The objective looks for `heldout_68_nme` first, then other overall 68 NME aliases.

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

For planner and CI smoke tests, `--mock-metrics` writes deterministic synthetic metrics instead of launching training:

```bash
python tools/landmarks/tune_training_hyperparameters.py \
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
- `optuna_study.json`
- `ranked_loss_candidates.json`
- `loss_finalist_summary.json`
- `ranked_lr_candidates.json`
- `lr_finalist_summary.json`
- `best_training_hyperparameters.json`
- `runs/<run-id>/config.json`
- `runs/<run-id>/command.txt`
- `runs/<run-id>/metrics.json`
- `runs/<run-id>/result.json`

The final JSON contains the selected values for:

- `star_loss_weight`
- `schema_consistency_weight`
- `auxiliary_loss_weight`
- `locw`
- `hw`
- `lr`

It also includes the exact training flags to reuse for the selected configuration.
