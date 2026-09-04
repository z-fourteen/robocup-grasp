from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .acceptance import run_acceptance
from .bop import import_bop_sequence
from .capture import capture_sequence_from_manifest
from .config import load_acceptance_config, load_reconstruction_config
from .errors import RobotGraspError, ValidationError
from .foundationpose_evaluation import evaluate_foundationpose_sequence
from .grasps import (
    add_candidate,
    delete_candidate,
    empty_grasps,
    get_candidate,
    load_grasps,
    make_candidate,
    save_grasps,
    update_candidate,
)
from .housecat6d import import_housecat6d_sequence
from .interfaces import GraspSelector, JsonCameraPoseProvider, JsonGraspCandidateProvider, JsonObjectPoseEstimator
from .object_frame import load_named_transform, set_object_frame
from .reconstruction import reconstruct_sequence
from .sequence import sequence_summary, validate_sequence
from .transforms import compose_base_grasp
from .viewer import run_grasp_viewer


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))


def _add_overwrite(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty output directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robot-grasp")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-sequence", help="Strictly validate an RGB-D sequence")
    validate.add_argument("--input", required=True, type=Path)
    validate.add_argument("--min-valid-depth-ratio", type=float, default=0.01)
    validate_mask = validate.add_mutually_exclusive_group()
    validate_mask.add_argument("--use-mask", dest="use_mask", action="store_true", default=True)
    validate_mask.add_argument("--no-mask", dest="use_mask", action="store_false")
    validate.add_argument(
        "--require-capture-metadata",
        action="store_true",
        help="Require real-capture hardware, timestamp, synchronization, and pose provenance",
    )

    capture = commands.add_parser(
        "capture-sequence",
        help="Convert a camera/robot JSONL manifest into a validated strict RGB-D sequence",
    )
    capture.add_argument("--manifest", required=True, type=Path, help="Normalized JSONL frame manifest")
    capture.add_argument("--config", required=True, type=Path, help="Capture hardware and camera profile YAML")
    capture.add_argument("--output", required=True, type=Path)
    _add_overwrite(capture)

    reconstruct = commands.add_parser("reconstruct", help="Fuse a validated sequence using Open3D TSDF")
    reconstruct.add_argument("--input", required=True, type=Path)
    reconstruct.add_argument("--output", required=True, type=Path)
    reconstruct.add_argument("--config", type=Path)
    _add_overwrite(reconstruct)

    object_frame = commands.add_parser("set-object-frame", help="Transform a model mesh into object coordinates")
    object_frame.add_argument("--mesh", required=True, type=Path)
    object_frame.add_argument("--transform", required=True, type=Path, help="JSON matrix or object containing T_object_model")
    object_frame.add_argument("--output", required=True, type=Path)
    object_frame.add_argument("--object-id", required=True)
    _add_overwrite(object_frame)

    grasps = commands.add_parser("grasps", help="Create, inspect, and edit grasps.json")
    grasp_commands = grasps.add_subparsers(dest="grasp_command", required=True)
    grasp_list = grasp_commands.add_parser("list")
    grasp_list.add_argument("--grasps", required=True, type=Path)
    grasp_show = grasp_commands.add_parser("show")
    grasp_show.add_argument("--grasps", required=True, type=Path)
    grasp_show.add_argument("--id", required=True)

    grasp_add = grasp_commands.add_parser("add")
    grasp_add.add_argument("--grasps", required=True, type=Path)
    grasp_add.add_argument("--object-id", help="Required only when creating a new grasps.json")
    grasp_add.add_argument("--id", required=True)
    grasp_add.add_argument("--transform", required=True, type=Path, help="JSON matrix or object containing T_object_grasp")
    grasp_add.add_argument("--pregrasp-offset", nargs=3, type=float)
    grasp_add.add_argument("--gripper-width", type=float, default=0.0)
    grasp_add.add_argument("--approach-distance", type=float, default=0.1)
    grasp_add.add_argument("--priority", type=int, default=0)
    grasp_add.add_argument("--disabled", action="store_true")
    grasp_add.add_argument("--symmetry-class", default="none")
    grasp_add.add_argument("--notes", default="")

    grasp_update = grasp_commands.add_parser("update")
    grasp_update.add_argument("--grasps", required=True, type=Path)
    grasp_update.add_argument("--id", required=True)
    grasp_update.add_argument("--transform", type=Path)
    grasp_update.add_argument("--pregrasp-offset", nargs=3, type=float)
    grasp_update.add_argument("--gripper-width", type=float)
    grasp_update.add_argument("--approach-distance", type=float)
    grasp_update.add_argument("--priority", type=int)
    enabled = grasp_update.add_mutually_exclusive_group()
    enabled.add_argument("--enabled", dest="enabled", action="store_true")
    enabled.add_argument("--disabled", dest="enabled", action="store_false")
    grasp_update.set_defaults(enabled=None)
    grasp_update.add_argument("--symmetry-class")
    grasp_update.add_argument("--notes")

    grasp_delete = grasp_commands.add_parser("delete")
    grasp_delete.add_argument("--grasps", required=True, type=Path)
    grasp_delete.add_argument("--id", required=True)

    viewer = commands.add_parser("view-grasps", help="Open the keyboard-driven Open3D annotation viewer")
    viewer.add_argument("--mesh", required=True, type=Path)
    viewer.add_argument("--grasps", required=True, type=Path)
    viewer.add_argument("--translation-step", type=float, default=0.002, help="Translation step in meters")
    viewer.add_argument("--rotation-step", type=float, default=2.0, help="Rotation step in degrees")

    compose = commands.add_parser("compose-grasp", help="Select and compose a grasp without a robot SDK")
    compose.add_argument("--camera-pose", required=True, type=Path)
    compose.add_argument("--object-pose", required=True, type=Path)
    compose.add_argument("--grasps", required=True, type=Path)

    acceptance = commands.add_parser("accept", help="Run mesh-dimension and pose-repeatability acceptance")
    acceptance.add_argument("--mesh", required=True, type=Path)
    acceptance.add_argument("--caliper", required=True, type=Path)
    acceptance.add_argument("--poses", required=True, type=Path)
    acceptance.add_argument("--output", required=True, type=Path)
    acceptance.add_argument("--config", type=Path)
    _add_overwrite(acceptance)

    bop = commands.add_parser("import-bop", help="Convert a BOP scene/object to the strict RGB-D sequence format")
    bop.add_argument("--dataset", required=True, type=Path, help="Extracted BOP dataset root, e.g. .../lm")
    bop.add_argument("--split", default="test")
    bop.add_argument("--scene", required=True, type=int)
    bop.add_argument("--object-id", required=True, type=int)
    bop.add_argument("--output", required=True, type=Path)
    bop.add_argument("--frame-step", type=int, default=1)
    bop.add_argument("--max-frames", type=int)
    _add_overwrite(bop)

    housecat = commands.add_parser(
        "import-housecat6d", help="Convert one HouseCat6D object track to the strict RGB-D sequence format"
    )
    housecat.add_argument("--dataset", required=True, type=Path, help="Extracted HouseCat6D dataset root")
    housecat.add_argument("--scene", required=True, help="Scene directory, e.g. val_scene1")
    housecat.add_argument("--object", required=True, dest="object_name", help="Exact HouseCat6D model name")
    housecat.add_argument("--output", required=True, type=Path)
    housecat.add_argument("--depth-source", choices=("depth", "depth_gt"), default="depth")
    housecat.add_argument("--frame-step", type=int, default=1)
    housecat.add_argument("--max-frames", type=int)
    housecat.add_argument("--min-mask-pixels", type=int, default=64)
    housecat.add_argument("--mask-erosion-pixels", type=int, default=0)
    _add_overwrite(housecat)

    fp_evaluate = commands.add_parser(
        "evaluate-foundationpose",
        help="Run FoundationPose on a strict sequence and compare predictions against held-back GT poses",
    )
    fp_evaluate.add_argument("--config", required=True, type=Path)
    fp_evaluate.add_argument("--sequence", required=True, type=Path)
    fp_evaluate.add_argument("--object-frame", required=True, type=Path)
    fp_evaluate.add_argument("--output", required=True, type=Path)
    fp_evaluate.add_argument("--frame-step", type=int, default=1)
    fp_evaluate.add_argument("--max-frames", type=int)
    _add_overwrite(fp_evaluate)
    return parser


def _run_grasps(args: argparse.Namespace) -> int:
    if args.grasp_command == "list":
        data = load_grasps(args.grasps)
        _print_json({"object_id": data["object_id"], "candidates": data["candidates"]})
        return 0
    if args.grasp_command == "show":
        _print_json(get_candidate(load_grasps(args.grasps), args.id))
        return 0
    if args.grasp_command == "add":
        if args.grasps.exists():
            data = load_grasps(args.grasps)
        elif args.object_id:
            data = empty_grasps(args.object_id)
        else:
            raise ValidationError(f"{args.grasps} does not exist. Pass --object-id to create it.")
        transform = load_named_transform(args.transform, "T_object_grasp")
        candidate = make_candidate(
            args.id,
            transform,
            pregrasp_offset=args.pregrasp_offset,
            gripper_width=args.gripper_width,
            approach_distance=args.approach_distance,
            priority=args.priority,
            enabled=not args.disabled,
            symmetry_class=args.symmetry_class,
            notes=args.notes,
        )
        data = add_candidate(data, candidate)
        save_grasps(args.grasps, data)
        _print_json(candidate)
        return 0
    if args.grasp_command == "update":
        data = load_grasps(args.grasps)
        changes = {}
        for field in ("pregrasp_offset", "gripper_width", "approach_distance", "priority", "enabled", "symmetry_class", "notes"):
            value = getattr(args, field)
            if value is not None:
                changes[field] = value
        if args.transform is not None:
            changes["T_object_grasp"] = load_named_transform(args.transform, "T_object_grasp").tolist()
        if not changes:
            raise ValidationError("No update fields were supplied. Pass a field such as --priority or --transform.")
        data = update_candidate(data, args.id, changes)
        save_grasps(args.grasps, data)
        _print_json(get_candidate(data, args.id))
        return 0
    if args.grasp_command == "delete":
        data = delete_candidate(load_grasps(args.grasps), args.id)
        save_grasps(args.grasps, data)
        _print_json({"deleted": args.id, "remaining": len(data["candidates"])})
        return 0
    raise AssertionError(args.grasp_command)


def run(args: argparse.Namespace) -> int:
    if args.command == "validate-sequence":
        sequence = validate_sequence(
            args.input,
            min_valid_depth_ratio=args.min_valid_depth_ratio,
            use_mask=args.use_mask,
            require_capture_metadata=args.require_capture_metadata,
        )
        _print_json(sequence_summary(sequence))
        return 0
    if args.command == "capture-sequence":
        _print_json(capture_sequence_from_manifest(
            args.manifest,
            args.config,
            args.output,
            overwrite=args.overwrite,
        ))
        return 0
    if args.command == "reconstruct":
        _print_json(reconstruct_sequence(args.input, args.output, load_reconstruction_config(args.config), overwrite=args.overwrite))
        return 0
    if args.command == "set-object-frame":
        _print_json(set_object_frame(args.mesh, args.transform, args.output, object_id=args.object_id, overwrite=args.overwrite))
        return 0
    if args.command == "grasps":
        return _run_grasps(args)
    if args.command == "view-grasps":
        run_grasp_viewer(args.mesh, args.grasps, translation_step_m=args.translation_step, rotation_step_deg=args.rotation_step)
        return 0
    if args.command == "compose-grasp":
        T_base_camera = JsonCameraPoseProvider(args.camera_pose).get_T_base_camera()
        T_camera_object = JsonObjectPoseEstimator(args.object_pose).estimate()
        selected = GraspSelector().select(JsonGraspCandidateProvider(args.grasps).get_candidates())
        T_base_grasp = compose_base_grasp(T_base_camera, T_camera_object, selected["T_object_grasp"])
        _print_json({
            "selected_grasp_id": selected["id"],
            "T_base_grasp": T_base_grasp.tolist(),
            "length_unit": "meter",
            "composition": "T_base_camera @ T_camera_object @ T_object_grasp",
        })
        return 0
    if args.command == "accept":
        report = run_acceptance(
            args.mesh,
            args.caliper,
            args.poses,
            args.output,
            load_acceptance_config(args.config),
            overwrite=args.overwrite,
        )
        _print_json(report)
        return 0 if report["passed"] else 1
    if args.command == "import-bop":
        _print_json(import_bop_sequence(
            args.dataset,
            args.output,
            split=args.split,
            scene_id=args.scene,
            object_id=args.object_id,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        ))
        return 0
    if args.command == "import-housecat6d":
        _print_json(import_housecat6d_sequence(
            args.dataset,
            args.output,
            scene_name=args.scene,
            object_name=args.object_name,
            depth_source=args.depth_source,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            min_mask_pixels=args.min_mask_pixels,
            mask_erosion_pixels=args.mask_erosion_pixels,
            overwrite=args.overwrite,
        ))
        return 0
    if args.command == "evaluate-foundationpose":
        _print_json(evaluate_foundationpose_sequence(
            args.config,
            args.sequence,
            args.object_frame,
            args.output,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        ))
        return 0
    raise AssertionError(args.command)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        return run(parser.parse_args(argv))
    except RobotGraspError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
