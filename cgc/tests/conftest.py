"""CGC unit test suite — shared fixtures and utilities."""

from __future__ import annotations

import os
import sys

import numpy as np

# Ensure cgc is importable even when running tests from their dir
_CGC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CGC_ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(_CGC_ROOT))

# ── shared constants (matching prod) ──────────────────────
M_P = 2.435300e18  # GeV — reduced Planck mass
M_CURV = 9.980738e17  # GeV — curvature scale
L_RP3 = 2.44  # dimensionless RP3 size at M_G
T_FLAVOR = 5  # Cartan index
ALPHA = 0.02  # Holst parameter
G3_MG = 0.496  # SU(3) gauge coupling at M_G
N_C = 3  # number of colors

# ── tolerance levels ─────────────────────────────────────
RTOL_NUMERICAL = 1e-10  # for float comparison of deterministic computation
RTOL_INDEPENDENT = 1e-8  # for cross-module consistency checks
RTOL_STABILITY = 5e-4  # for convergence/stability checks (5e-4 ~ 2x grid coarsening error)
ATOL = 1e-30  # negligible absolute floor

# ── k-grid for trace density tests ───────────────────────
DEFAULT_K_BINS = 500


def assert_close(a: float, b: float, rtol: float = RTOL_NUMERICAL, label: str = "") -> bool:
    """Assert two floats are equal within relative tolerance."""
    if b == 0.0:
        ok = abs(a) < rtol
    elif abs(b) == float("inf"):
        ok = abs(a) == float("inf")
    else:
        ok = abs(a - b) / max(abs(b), ATOL) < rtol
    if not ok:
        msg = f"  MISMATCH [{label}]: {a:.12e} != {b:.12e}"
        print(msg)
    return ok


def k_grid(n_bins: int = DEFAULT_K_BINS) -> np.ndarray:
    """Log-spaced k grid from 1 GeV to M_P."""
    return np.geomspace(1.0, M_P, n_bins)
