from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .errors import ValidationError


def load_json(path: str | Path) -> Any:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ValidationError(f"Missing JSON file: {path}. Create the file at this path.") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}."
        ) from exc


def dump_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def load_yaml(path: str | Path) -> Any:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ValidationError(f"Missing YAML file: {path}. Create it or pass the correct --config path.") from exc
    except yaml.YAMLError as exc:
        raise ValidationError(f"Invalid YAML in {path}: {exc}.") from exc
    return {} if data is None else data


def dump_yaml(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True, default_flow_style=False)


def prepare_output_dir(path: str | Path, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise ValidationError(f"Output path {path} exists and is not a directory. Choose another path.")
        if not overwrite:
            raise ValidationError(
                f"Output directory {path} already exists. "
                "Choose a new directory or pass --overwrite."
            )
    else:
        path.mkdir(parents=True)
    return path
