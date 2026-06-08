"""Config loading and argparse merge helpers for landmark pipeline CLIs.

Merge semantics:
  * Config values are converted to CLI tokens and prepended before explicit CLI
    tokens. That keeps normal scalar argparse behavior: the later user-provided
    CLI value wins.
  * argparse append-style options intentionally merge. For example, config
    ``train_arg`` entries are followed by any CLI ``--train-arg`` entries, and
    config ``dataset_sources`` entries are followed by CLI ``--dataset-source``
    entries. This is useful for adding one-off trainer flags or dataset source
    overrides without copying the whole config.
  * Unknown config keys fail fast. Trainer-only options belong in ``train_arg``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import typing as T
from pathlib import Path


APPEND_MERGE_ACTION_CLASS_NAMES = {"_AppendAction"}


def _extract_config_path(argv: T.Sequence[str]) -> Path | None:
    for index, token in enumerate(argv):
        if token == "--config" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if token.startswith("--config="):
            return Path(token.split("=", 1)[1])
    return None


def _load_pipeline_config(path: Path) -> dict[str, T.Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing pipeline config: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "YAML configs require PyYAML. Use JSON or install pyyaml."
            ) from exc
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(
            f"unsupported config extension {suffix!r}; use .json, .yaml, or .yml"
        )

    if not isinstance(payload, dict):
        raise ValueError(f"pipeline config must be a JSON/YAML object: {path}")
    return payload


def _normalize_config_key(key: str) -> str:
    return str(key).strip().replace("-", "_")


def _put_config_value(target: dict[str, T.Any], key: str, value: T.Any) -> None:
    key = _normalize_config_key(key)
    if key in {"config", "description", "notes", "comment"}:
        return
    if key in target and isinstance(target[key], list) and isinstance(value, list):
        target[key].extend(value)
    else:
        target[key] = value


def _hard_negative_extra_arg(key: str, value: T.Any) -> list[str]:
    flag = "--" + _normalize_config_key(key).replace("_", "-")
    if isinstance(value, bool):
        return [flag] if value else []
    if isinstance(value, (list, tuple)):
        return [f"{flag} {shlex.quote(str(item))}" for item in value]
    if isinstance(value, dict):
        return [f"{flag} {shlex.quote(json.dumps(value, sort_keys=True))}"]
    if value is None:
        return []
    return [f"{flag} {shlex.quote(str(value))}"]


def _flatten_pipeline_config(config: T.Mapping[str, T.Any]) -> dict[str, T.Any]:
    flat: dict[str, T.Any] = {}

    for raw_key, value in config.items():
        key = _normalize_config_key(raw_key)

        if key in {"description", "notes", "comment"}:
            continue

        if key == "datasets":
            _put_config_value(flat, "dataset", value)
            continue

        if key in {"dataset_sources", "dataset_source_map"}:
            _put_config_value(flat, "dataset_source", value)
            continue

        if key in {"dataset_source_zips", "dataset_source_zip_map"}:
            _put_config_value(flat, "dataset_source_zip", value)
            continue

        if key in {"training", "runtime", "eval", "checkpoint"}:
            if not isinstance(value, dict):
                raise ValueError(f"config section {raw_key!r} must be an object")
            for section_key, section_value in value.items():
                _put_config_value(flat, section_key, section_value)
            continue

        if key == "hard_negative":
            if not isinstance(value, dict):
                raise ValueError("config section 'hard_negative' must be an object")
            extra_args: list[str] = []
            for section_key, section_value in value.items():
                section_key = _normalize_config_key(section_key)
                if section_key in {
                    "max_profile_occlusion",
                    "max_profile",
                    "max_occlusion",
                    "max_anchors",
                    "exclude_image_ids_file",
                }:
                    _put_config_value(flat, section_key, section_value)
                elif section_key in {"arg", "args", "extra_args"}:
                    if isinstance(section_value, list):
                        extra_args.extend(str(item) for item in section_value)
                    else:
                        extra_args.append(str(section_value))
                else:
                    extra_args.extend(_hard_negative_extra_arg(section_key, section_value))
            if extra_args:
                existing = flat.get("hard_negative_arg", [])
                if not isinstance(existing, list):
                    existing = [existing]
                flat["hard_negative_arg"] = [*existing, *extra_args]
            continue

        _put_config_value(flat, key, value)

    return flat


def _parser_action_by_dest(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    actions: dict[str, argparse.Action] = {}
    for action in parser._actions:  # argparse exposes no public dest index.
        if not action.option_strings or action.dest in {"help", argparse.SUPPRESS}:
            continue
        actions[_normalize_config_key(action.dest)] = action
    return actions


def _positive_option(action: argparse.Action) -> str:
    for option in action.option_strings:
        if option.startswith("--") and not option.startswith("--no-"):
            return option
    return action.option_strings[0]


def _negative_option(action: argparse.Action) -> str | None:
    for option in action.option_strings:
        if option.startswith("--no-"):
            return option
    return None


def _is_append_action(action: argparse.Action) -> bool:
    return action.__class__.__name__ in APPEND_MERGE_ACTION_CLASS_NAMES


def _append_config_cli_value(
    argv: list[str],
    action: argparse.Action,
    key: str,
    value: T.Any,
) -> None:
    if value is None:
        return

    option = _positive_option(action)

    if isinstance(value, bool):
        negative = _negative_option(action)
        if isinstance(action, argparse.BooleanOptionalAction):
            argv.append(option if value else negative or option)
        elif value:
            argv.append(option)
        elif negative:
            argv.append(negative)
        return

    if key == "dataset" and isinstance(value, (list, tuple)):
        argv.extend([option, ",".join(str(item) for item in value)])
        return

    if key in {"dataset_source", "dataset_source_zip"} and isinstance(value, dict):
        for dataset, path in value.items():
            argv.extend([option, f"{dataset}={path}"])
        return

    if _is_append_action(action):
        items = value if isinstance(value, list) else [value]
        for item in items:
            # Use --option=value so values that themselves begin with "--" are
            # parsed as values rather than mistaken for top-level CLI options.
            argv.append(f"{option}={item}")
        return

    if isinstance(value, (list, tuple)):
        argv.extend([option, ",".join(str(item) for item in value)])
        return

    if isinstance(value, dict):
        argv.extend([option, json.dumps(value, sort_keys=True)])
        return

    argv.extend([option, str(value)])


def _config_to_argv(
    parser: argparse.ArgumentParser,
    config: T.Mapping[str, T.Any],
) -> list[str]:
    actions = _parser_action_by_dest(parser)
    flat = _flatten_pipeline_config(config)
    argv: list[str] = []

    unknown = sorted(key for key in flat if key not in actions)
    if unknown:
        raise ValueError(
            "config contains unknown pipeline option(s): "
            + ", ".join(unknown)
            + ". Use train_arg for TrainHeatmapStageFP16.py-only options."
        )

    for key, value in flat.items():
        _append_config_cli_value(argv, actions[key], key, value)

    return argv


def _merge_config_argv(
    parser: argparse.ArgumentParser,
    config_path: Path | None,
    cli_argv: T.Sequence[str],
) -> list[str]:
    if config_path is None:
        return list(cli_argv)
    config = _load_pipeline_config(config_path)

    # Config args come first. Scalar CLI args override because argparse keeps the
    # later value. Append actions intentionally merge config entries with CLI
    # entries in that order.
    return [*_config_to_argv(parser, config), *cli_argv]


def _json_safe_pipeline_value(value: T.Any) -> T.Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_pipeline_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_safe_pipeline_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    return str(value)
