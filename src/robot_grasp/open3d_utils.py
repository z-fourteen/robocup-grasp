from __future__ import annotations

from .errors import OptionalDependencyError


def require_open3d(feature: str):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise OptionalDependencyError(
            f"{feature} requires Open3D, but it is not installed in this Python environment. "
            "Install it with 'python -m pip install open3d' (or the version approved for your deployment)."
        ) from exc
    return o3d
