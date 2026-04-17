"""
Affine registration from manually placed landmarks.

Placeholder — implementation deferred to the next commit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from skimage.transform import AffineTransform


def compute_affine_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> tuple[AffineTransform, float, np.ndarray]:
    """Fit an affine transform from src_points → dst_points.

    Parameters
    ----------
    src_points:
        (n ≥ 3, 2) landmark coordinates in the *moving* frame.
    dst_points:
        (n ≥ 3, 2) matching landmark coordinates in the *fixed* frame.

    Returns
    -------
    transform:
        Fitted skimage AffineTransform.
    rms_error:
        Root-mean-square residual distance (in dst units).
    residuals:
        Per-landmark residual distances (n,).
    """
    if len(src_points) < 3:
        raise ValueError("At least 3 landmark pairs are required.")

    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)

    # Use the current skimage ≥0.26 class constructor (estimate() is deprecated)
    transform = AffineTransform.from_estimate(src, dst)

    projected = transform(src)  # apply transform to src points
    diffs = projected - dst
    residuals = np.linalg.norm(diffs, axis=1)
    rms_error = float(np.sqrt(np.mean(residuals ** 2)))

    return transform, rms_error, residuals


def apply_transform(
    points: np.ndarray,
    transform: AffineTransform,
) -> np.ndarray:
    """Apply an affine transform to a set of 2D points.

    Parameters
    ----------
    points:
        (n, 2) array of coordinates to transform.
    transform:
        Fitted skimage AffineTransform.

    Returns
    -------
    (n, 2) transformed coordinates as float32.
    """
    pts = np.asarray(points, dtype=np.float64)
    return transform(pts).astype(np.float32)


def _make_transform_dict(
    transform: AffineTransform,
    rms_error_um: float = 0.0,
    n_landmarks: int = 0,
) -> dict:
    """Build the canonical transform dict used by save_transform and the UI callbacks."""
    return {
        "matrix": transform.params.tolist(),
        "rms_error_um": float(rms_error_um),
        "n_landmarks": int(n_landmarks),
        "description": (
            "Affine matrix mapping MEA (moving) → Slide-tags (fixed). "
            "Apply directly to MEA unit coordinates to place them in Slide-tags space."
        ),
    }


def save_transform(
    transform: AffineTransform,
    path: str | Path,
    rms_error_um: float = 0.0,
    n_landmarks: int = 0,
) -> None:
    """Save a transform to a JSON file.

    The stored matrix maps MEA (moving) → Slide-tags (fixed).
    Apply it directly to MEA unit coordinates.
    """
    data = _make_transform_dict(transform, rms_error_um=rms_error_um,
                                n_landmarks=n_landmarks)
    Path(path).write_text(json.dumps(data, indent=2))


def load_transform(path: str | Path) -> AffineTransform:
    """Load a transform from a JSON file saved by save_transform()."""
    data = json.loads(Path(path).read_text())
    matrix = np.array(data["matrix"])
    t = AffineTransform(matrix=matrix)
    return t
