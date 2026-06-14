# Landmark auxiliary and visibility supervision

This document records the training contract used by issue #4.

## Auxiliary labels

Auxiliary labels are supervised only when explicit labels are present in one of:

- `sample["auxiliary_labels"][task]`
- `metadata["auxiliary_labels"][task]`
- an explicit task field on the sample or metadata
- `metadata["attributes"][task]`

Missing labels stay `-1` and are skipped. Collate does not infer clean negatives from absent metadata. In particular:

- missing occlusion is not `no_occlusion`
- landmark masks are not treated as visibility truth
- sample weights are not treated as landmark confidence truth

Each label keeps provenance in `batch["aux_provenance"]`.

## Per-point visibility

Per-point visibility targets use:

```text
1 = visible
0 = occluded
-1 = unknown / skipped
```

Visibility heads are schema-specific:

landmarks_68  -> visibility_68
landmarks_98  -> visibility_98
landmarks_106 -> visibility_106
landmarks_194 -> visibility_194
landmarks_29  -> visibility_29
profile39     -> visibility_profile39

The visibility loss is masked BCE. Missing labels are never trained as visible.
Explicit visibility labels are independent from the landmark localization mask:
an occluded point can be excluded from coordinate/heatmap loss while still
providing a valid negative target to the visibility head.

Warm start schedule

Visibility loss is controlled by:

--visibility-loss-weight
--visibility-loss-initial-weight
--visibility-loss-start-epoch
--visibility-loss-ramp-epochs

Default --visibility-loss-weight 0.0 keeps visibility off until explicitly enabled.

Synthetic occlusion pseudo-labels

synthetic_visibility_from_occluder_mask() can generate pseudo visibility labels from an occluder mask. Manifest entries may provide `occluder_mask` or `synthetic_occluder_mask`; when explicit visibility labels are absent, the loader converts that mask into per-point pseudo visibility with provenance `synthetic_occluder_mask`. Pseudo labels are weighted separately through `--visibility-pseudo-loss-weight` and are disabled by default when that value is `0.0`.

Optional point/edge maps

OccFace-style point and edge maps remain a future lower-priority extension. They should be disabled by default and added only after per-point visibility is stable.
