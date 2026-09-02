from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .errors import ValidationError
from .io_utils import dump_json, load_json
from .transforms import validate_transform


SCHEMA_PATH = Path(__file__).with_name("grasps.schema.json")
TRANSFORM_CONVENTION = "T_dst_src transforms points from src coordinates into dst coordinates"


def empty_grasps(object_id: str) -> dict[str, Any]:
    if not isinstance(object_id, str) or not object_id.strip():
        raise ValidationError("object_id must be a non-empty string when creating grasps.json.")
    return {
        "schema_version": 1,
        "object_id": object_id,
        "length_unit": "meter",
        "transform_convention": TRANSFORM_CONVENTION,
        "candidates": [],
    }


def validate_grasps(data: Any, *, source: str = "grasps data") -> dict[str, Any]:
    schema = load_json(SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda error: list(error.absolute_path))
    if errors:
        details = []
        for error in errors:
            location = ".".join(str(item) for item in error.absolute_path) or "root"
            details.append(f"{source}:{location}: {error.message}")
        raise ValidationError("Grasp schema validation failed:\n- " + "\n- ".join(details))
    ids: set[str] = set()
    for index, candidate in enumerate(data["candidates"]):
        candidate_id = candidate["id"]
        if candidate_id in ids:
            raise ValidationError(f"{source} has duplicate grasp id '{candidate_id}'. Rename or delete one candidate.")
        ids.add(candidate_id)
        validate_transform(candidate["T_object_grasp"], name=f"T_object_grasp for '{candidate_id}' in {source}")
        lengths = [*candidate["pregrasp_offset"], candidate["gripper_width"], candidate["approach_distance"]]
        if not all(math.isfinite(float(value)) for value in lengths):
            raise ValidationError(
                f"{source} candidate '{candidate_id}' contains NaN or infinity in a length field; use finite meters."
            )
    return data


def load_grasps(path: str | Path) -> dict[str, Any]:
    return validate_grasps(load_json(path), source=str(path))


def save_grasps(path: str | Path, data: dict[str, Any]) -> None:
    validate_grasps(data, source=str(path))
    dump_json(path, data)


def make_candidate(
    candidate_id: str,
    T_object_grasp: Any,
    *,
    pregrasp_offset: list[float] | None = None,
    gripper_width: float = 0.0,
    approach_distance: float = 0.1,
    priority: int = 0,
    enabled: bool = True,
    symmetry_class: str = "none",
    notes: str = "",
) -> dict[str, Any]:
    transform = validate_transform(T_object_grasp, name=f"T_object_grasp for '{candidate_id}'")
    candidate = {
        "id": candidate_id,
        "T_object_grasp": transform.tolist(),
        "pregrasp_offset": [0.0, 0.0, -approach_distance] if pregrasp_offset is None else pregrasp_offset,
        "gripper_width": gripper_width,
        "approach_distance": approach_distance,
        "priority": priority,
        "enabled": enabled,
        "symmetry_class": symmetry_class,
        "notes": notes,
    }
    validate_grasps(empty_grasps("validation") | {"candidates": [candidate]}, source=f"candidate '{candidate_id}'")
    return candidate


def get_candidate(data: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in data["candidates"]:
        if candidate["id"] == candidate_id:
            return candidate
    raise ValidationError(f"No grasp candidate with id '{candidate_id}'. Use 'grasps list' to inspect available ids.")


def add_candidate(data: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(data)
    if any(item["id"] == candidate["id"] for item in result["candidates"]):
        raise ValidationError(f"Grasp id '{candidate['id']}' already exists. Use 'grasps update' or choose another id.")
    result["candidates"].append(candidate)
    return validate_grasps(result)


def update_candidate(data: dict[str, Any], candidate_id: str, changes: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(data)
    candidate = get_candidate(result, candidate_id)
    unknown = sorted(set(changes) - set(candidate))
    if unknown:
        raise ValidationError(f"Unknown candidate fields {unknown}; allowed fields are {sorted(candidate)}.")
    candidate.update(changes)
    return validate_grasps(result)


def delete_candidate(data: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    result = deepcopy(data)
    get_candidate(result, candidate_id)
    result["candidates"] = [item for item in result["candidates"] if item["id"] != candidate_id]
    return validate_grasps(result)
