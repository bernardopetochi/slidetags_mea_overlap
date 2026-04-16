"""
Tests for affine registration.

Uses a known synthetic transform (translation + rotation) and verifies
that compute_affine_transform() recovers it within floating-point tolerance.
"""

import numpy as np
import pytest
from skimage.transform import AffineTransform

from analysis.registration import compute_affine_transform, apply_transform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_landmarks(n: int = 6, seed: int = 42):
    """Generate n landmark pairs related by a known affine transform."""
    rng = np.random.default_rng(seed)

    # Ground-truth transform: 15° rotation + translation (30, -20)
    angle = np.deg2rad(15)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    matrix = np.array([
        [cos_a, -sin_a, 30.0],
        [sin_a,  cos_a, -20.0],
        [0.0,    0.0,    1.0],
    ])
    true_transform = AffineTransform(matrix=matrix)

    src = rng.uniform(0, 500, size=(n, 2)).astype(np.float64)
    dst = true_transform(src)
    return src, dst, true_transform


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_recover_known_transform():
    """Fitted transform should reproduce the known matrix to ≤1e-4 tolerance."""
    src, dst, true_t = _make_landmarks(n=8)

    fitted, rms, residuals = compute_affine_transform(src, dst)

    np.testing.assert_allclose(
        fitted.params, true_t.params, atol=1e-4,
        err_msg="Recovered matrix differs from ground truth",
    )


def test_rms_error_near_zero_on_exact_data():
    """RMS error must be negligible on noise-free landmark pairs."""
    src, dst, _ = _make_landmarks(n=6)
    _, rms, residuals = compute_affine_transform(src, dst)

    assert rms < 1e-6, f"Expected near-zero RMS, got {rms}"
    assert np.all(residuals < 1e-6), "Some per-landmark residuals are non-zero"


def test_minimum_landmarks_enforced():
    """Fewer than 3 landmark pairs must raise ValueError."""
    src = np.array([[0, 0], [1, 0]], dtype=np.float64)
    dst = np.array([[0, 0], [1, 0]], dtype=np.float64)

    with pytest.raises(ValueError, match="3 landmark"):
        compute_affine_transform(src, dst)


def test_apply_transform_shape():
    """apply_transform must preserve input shape and return float32."""
    src, dst, _ = _make_landmarks(n=4)
    transform, _, _ = compute_affine_transform(src, dst)

    points = np.random.default_rng(0).uniform(0, 100, (50, 2))
    result = apply_transform(points, transform)

    assert result.shape == (50, 2)
    assert result.dtype == np.float32
