#!/usr/bin/env python3
"""Validate a schema-aware landmark training manifest."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.manifest.validator import validate_training_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--allow-legacy-68-projection", action="store_true")
    parser.add_argument("--allow-missing-projection-audit", action="store_true")
    parser.add_argument(
        "--allow-legacy-missing-contract-fields",
        action="store_true",
        help="Do not fail legacy manifests missing inferable contract fields: landmark_count, head_name, split_safe_id.",
    )
    parser.add_argument("--skip-image-exists-check", action="store_true")
    parser.add_argument("--max-examples", type=int, default=25)
    parser.add_argument(
        "--geometry-overlay-dir",
        type=Path,
        default=None,
        help=(
            "Write review overlay PNGs (loader-scaled landmarks over the 256 "
            "crop) for samples flagged by the geometry checks."
        ),
    )
    parser.add_argument("--max-geometry-overlays", type=int, default=200)
    args = parser.parse_args()

    report = validate_training_manifest(
        args.manifest,
        report_path=args.report,
        require_images=not args.skip_image_exists_check,
        allow_legacy_68_projection=args.allow_legacy_68_projection,
        allow_missing_projection_audit=args.allow_missing_projection_audit,
        allow_legacy_missing_contract_fields=args.allow_legacy_missing_contract_fields,
        max_examples=args.max_examples,
        raise_on_error=False,
        geometry_overlay_dir=args.geometry_overlay_dir,
        max_geometry_overlays=args.max_geometry_overlays,
    )
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "manifest": report["manifest"],
                "total_samples": report["total_samples"],
                "valid_samples": report["valid_samples"],
                "invalid_samples": report["invalid_samples"],
                "schemas": report["schemas"],
                "heads": report["heads"],
                "leakage_violations": report["leakage"]["violation_count"],
                "geometry": report["geometry"],
                "report": str(args.report) if args.report else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl-C).", file=sys.stderr)
        raise SystemExit(130)
