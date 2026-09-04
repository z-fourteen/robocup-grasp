"""Optional Zenoh JSON recorder for the RPP robot interfaces."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterable

from .errors import ValidationError


@dataclass(frozen=True)
class ZenohTopicSet:
    device_group_name: str
    side: str = "right"

    def __post_init__(self) -> None:
        if not self.device_group_name.strip():
            raise ValidationError("device_group_name must be non-empty.")
        if self.side not in {"left", "right"}:
            raise ValidationError("Zenoh arm side must be left or right.")

    @property
    def joint_angle(self) -> str:
        return f"{self.device_group_name}/slave_arm/{self.side}/Joint_angle"

    @property
    def relative_pose(self) -> str:
        return f"{self.device_group_name}/slave_arm/{self.side}/Relative_pose"

    @property
    def gripper_status(self) -> str:
        return f"{self.device_group_name}/slave_arm/{self.side}/Gripper_status"


def _payload_bytes(payload: Any) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode("utf-8")
    to_bytes = getattr(payload, "to_bytes", None)
    if callable(to_bytes):
        return bytes(to_bytes())
    raise ValidationError("Zenoh JSON payload must be bytes, str, or expose to_bytes().")


def decode_json_payload(payload: Any) -> dict[str, Any]:
    try:
        data = json.loads(_payload_bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("Zenoh payload is not valid UTF-8 JSON; fastcdr requires a separate decoder.") from exc
    if not isinstance(data, dict):
        raise ValidationError("Zenoh JSON payload must decode to an object.")
    return data


def _message_timestamp_ns(message: dict[str, Any], *, fallback_ns: int) -> int:
    header = message.get("header")
    stamp = header.get("stamp") if isinstance(header, dict) else None
    if isinstance(stamp, dict) and "sec" in stamp and "nanosec" in stamp:
        try:
            sec = int(stamp["sec"])
            nanosec = int(stamp["nanosec"])
        except (TypeError, ValueError) as exc:
            raise ValidationError("Zenoh header.stamp sec/nanosec must be integers.") from exc
        if sec < 0 or not 0 <= nanosec < 1_000_000_000:
            raise ValidationError("Zenoh header.stamp sec/nanosec is out of range.")
        return sec * 1_000_000_000 + nanosec
    timestamp = message.get("timestamp")
    if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
        if timestamp < 0:
            raise ValidationError("Zenoh timestamp must be non-negative.")
        # The VR JSON interface documents Unix milliseconds.
        return int(round(float(timestamp) * 1_000_000.0))
    return fallback_ns


class ZenohJsonRecorder:
    """Record selected JSON topics without imposing robot kinematics semantics."""

    def __init__(self, *, device_group_name: str, side: str = "right", zenoh_config: str | Path | None = None):
        self.topics = ZenohTopicSet(device_group_name, side)
        self.zenoh_config = Path(zenoh_config) if zenoh_config is not None else None

    def record(
        self,
        output: str | Path,
        *,
        duration_seconds: float = 10.0,
        topic_names: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        if duration_seconds <= 0:
            raise ValidationError("duration_seconds must be positive.")
        topics = tuple(topic_names or (self.topics.joint_angle, self.topics.relative_pose, self.topics.gripper_status))
        if not topics or any(not isinstance(topic, str) or not topic.strip() for topic in topics):
            raise ValidationError("At least one non-empty Zenoh topic is required.")
        try:
            import zenoh
        except ImportError as exc:
            raise ValidationError(
                "Python Zenoh SDK is unavailable. Install eclipse-zenoh or record the topics with the existing "
                "RPP process and convert its JSON output to this format."
            ) from exc
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        lock = threading.Lock()

        def callback(sample: Any) -> None:
            received_ns = time.time_ns()
            message = decode_json_payload(sample.payload)
            record = {
                "topic": str(sample.key_expr),
                "received_timestamp_ns": received_ns,
                "message_timestamp_ns": _message_timestamp_ns(message, fallback_ns=received_ns),
                "message": message,
            }
            with lock:
                records.append(record)

        config = zenoh.Config.from_file(str(self.zenoh_config)) if self.zenoh_config else zenoh.Config()
        session = zenoh.open(config)
        subscribers = []
        try:
            subscribers = [session.declare_subscriber(topic, callback) for topic in topics]
            time.sleep(duration_seconds)
        finally:
            for subscriber in subscribers:
                subscriber.undeclare()
            session.close()
        records.sort(key=lambda item: (item["message_timestamp_ns"], item["received_timestamp_ns"], item["topic"]))
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        report = {
            "status": "recorded",
            "topics": list(topics),
            "duration_seconds": float(duration_seconds),
            "message_count": len(records),
            "output": str(output_path),
        }
        return report
