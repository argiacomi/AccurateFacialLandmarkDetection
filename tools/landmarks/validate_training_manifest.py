#!/usr/bin/env python3
"""Validate a schema-aware landmark training manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.manifest.validator import validate_training_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--allow-legacy-68-projection", action="store_true")
    parser.add_argument("--allow-missing-projection-audit", action="store_true")
    parser.add_argument("--skip-image-exists-check", action="store_true")
    parser.add_argument("--max-examples", type=int, default=25)
    args = parser.parse_args()

    report = validate_training_manifest(
        args.manifest,
        report_path=args.report,
        require_images=not args.skip_image_exists_check,
        allow_legacy_68_projection=args.allow_legacy_68_projection,
        allow_missing_projection_audit=args.allow_missing_projection_audit,
        max_examples=args.max_examples,
        raise_on_error=False,
    )
    print(json.dumps({
        "ok": report["ok"],
        "manifest": report["manifest"],
        "total_samples": report["total_samples"],
        "valid_samples": report["valid_samples"],
        "invalid_samples": report["invalid_samples"],
        "schemas": report["schemas"],
        "heads": report["heads"],
        "leakage_violations": report["leakage"]["violation_count"],
        "report": str(args.report) if args.report else None,
    }, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
