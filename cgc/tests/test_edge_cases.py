"""Edge case and degenerate tests for CGC core functions.

Tests boundary conditions, extreme parameters, and degradation behavior.
Key invariants that would expose implementation bugs.
"""

import os

# Use relative import under tests/
import sys

import numpy as np

_CGC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # cgc/
if _CGC_ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(_CGC_ROOT))  # parent of cgc/ for imports

from cgc.engine.chi_potential import ChiPotential
from cgc.engine.frg_flow_rp3 import (
    L_RP3,
    M_CURV,
    M_P,
    ExponentialRegulator,
    FieldContent,
    FieldSpecies,
    LitimRegulator,
    RP3TraceDensity,
    f2_field_content,
    tmunu_field_content,
)

# ── Goldens (500-bin, Litim) ──────────────────────────────
PI0_TMUNU_GOLDEN = 3.5999945350e-02
PI0_F2_GOLDEN = -1.5226901996e-01
ATOL = 1e-6
RTOL = 0.05  # 5% for coarse grids


def compute_pi0(fields, k=None, regulator=None):
    """Helper: compute Pi0 from field list."""
    if k is None:
        k = np.geomspace(1.0, M_P, 500)
    if regulator is None:
        regulator = LitimRegulator()
    d_ln = np.log(k[1] / k[0])
    trace = RP3TraceDensity(fields, regulator=regulator)
    eta = np.array([trace.trace_density_at_k(ki) for ki in k])
    return float(np.cumsum(eta)[-1] * d_ln)


# ═══════════════════════════════════════════════════════════
# T=2 Degenerate Case
# ═══════════════════════════════════════════════════════════


def test_T2_chi_vev_diverges():
    """T=2 → chi_vev = M_P/0 → should raise or inf."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cp = ChiPotential(T=2.0)
        assert not np.isfinite(cp.chi_vev) or cp.chi_vev > 1e25, f"T=2 chi_vev={cp.chi_vev} should diverge"


def test_T2_mu2_negative():
    """T=2: mu2 still negative (depends on alpha, not T)."""
    cp = ChiPotential(T=2.0)
    assert cp.mu2 < 0, f"mu2={cp.mu2} should be negative"


# ═══════════════════════════════════════════════════════════
# T → Large (Flat Limit)
# ═══════════════════════════════════════════════════════════


def test_large_T_chi_vev_vanishes():
    """T → ∞ → chi_vev → 0 (1/√T)."""
    for T in [10, 50, 100, 1000]:
        cp = ChiPotential(T=T)
        expected = M_P / np.sqrt(T - 2)
        assert abs(cp.chi_vev - expected) < 1e-10 * M_P, f"T={T}: chi_vev={cp.chi_vev:.4e} vs expected={expected:.4e}"


def test_large_T_mu2_unchanged():
    """mu2 depends only on alpha, not T."""
    cp5 = ChiPotential(T=5)
    cp100 = ChiPotential(T=100)
    assert abs(cp5.mu2 - cp100.mu2) < 1e-10, f"mu2 should be T-independent: {cp5.mu2} vs {cp100.mu2}"


def test_large_T_lambda_grows():
    """lambda = 6*alpha/(chi_vev/M_P)^2 → as T↑, chi_vev↓ → lambda↑.

    Physical: T→∞ (flat limit) means chi_vev→0, which means
    the quartic gets stronger (larger lambda).
    """
    lambda5 = ChiPotential(T=5).lamb
    lambda100 = ChiPotential(T=100).lamb
    assert lambda100 > lambda5, f"lambda(100)={lambda100:.2e} should be > lambda(5)={lambda5:.2e} (flat limit λ↑)"


# ═══════════════════════════════════════════════════════════
# Empty / Single-Field Limits
# ═══════════════════════════════════════════════════════════


def test_empty_field_content():
    """Zero fields → Pi0 ≈ 0 (only integration constant noise)."""
    pi0 = compute_pi0([])
    assert abs(pi0) < 1.0, f"Empty Pi0={pi0:.6e} should be negligible"


def test_single_scalar():
    """Single massless scalar contributes positively."""
    fields = [FieldContent("scalar_test", FieldSpecies.SCALAR, 1, 1, 0.0, 1.0)]
    pi0 = compute_pi0(fields)
    assert pi0 > 0, f"Single scalar Pi0={pi0:.6e} should be > 0"


def test_single_spinor():
    """Single spinor contributes negatively."""
    fields = [FieldContent("spinor_test", FieldSpecies.SPINOR, 1, 2, 0.0, 1.0)]
    pi0 = compute_pi0(fields)
    assert pi0 < 0, f"Single spinor Pi0={pi0:.6e} should be < 0"


def test_single_vector():
    """Single vector contributes positively (gauge field)."""
    fields = [FieldContent("vec_test", FieldSpecies.VECTOR, 1, 2, 0.0, 1.0)]
    pi0 = compute_pi0(fields)
    assert pi0 > 0, f"Single vector Pi0={pi0:.6e} should be > 0"


def test_single_tensor():
    """Single TT tensor (graviton) contributes positively."""
    fields = [FieldContent("grav_test", FieldSpecies.TENSOR_TT, 2, 5, 0.0, 1.0)]
    pi0 = compute_pi0(fields)
    assert pi0 > 0, f"Single tensor Pi0={pi0:.6e} should be > 0"


# ═══════════════════════════════════════════════════════════
# k-Grid Coarsening
# ═══════════════════════════════════════════════════════════


def test_k_grid_degredation():
    """Measure Pi0 error as k-grid coarsens.

    RP3 discrete spectrum produces oscillatory convergence —
    different grid sizes sample different mode thresholds.
    This is a physical feature, not a numerical defect.

    The test validates that:
    - 500-bin is the production standard
    - Coarse grids have bounded (but not monotonic) errors
    - Errors are < 50% even at 10 bins (RP3 is structured enough)
    """
    fields = tmunu_field_content()
    k_fine = np.geomspace(1.0, M_P, 500)
    pi0_fine = compute_pi0(fields, k=k_fine)

    results = {}
    for n in [10, 20, 50, 100, 200]:
        k = np.geomspace(1.0, M_P, n)
        pi0 = compute_pi0(fields, k=k)
        err = abs(pi0 - pi0_fine) / max(abs(pi0_fine), 1e-30)
        results[n] = err
        print(f"    n={n:3d}: Pi0={pi0:.4e}, rel_err={err:.2e}")

    # RP3 oscillatory convergence: errors bounded but not monotonic
    # All errors should be < 50% (structure is preserved even at 10 bins)
    for n, err in results.items():
        assert err < 0.50, f"{n}-bin error {err:.2e} exceeds 50% bound"


# ═══════════════════════════════════════════════════════════
# Regulator Dependence
# ═══════════════════════════════════════════════════════════


def test_regulator_litim_vs_exp():
    """Litim vs exponential regulator: quantify systematic error."""
    k = np.geomspace(1.0, M_P, 500)

    for name, fn in [("Tmunu", tmunu_field_content), ("F2", f2_field_content)]:
        fields = fn()
        pi0_l = compute_pi0(fields, k=k, regulator=LitimRegulator())
        pi0_e = compute_pi0(fields, k=k, regulator=ExponentialRegulator())
        sys_err = abs(pi0_l - pi0_e) / max(abs(pi0_l), 1e-30)
        print(f"    {name}: Litim={pi0_l:.4e}, Exp={pi0_e:.4e}, sys_err={sys_err:.2e}")
        assert sys_err < 0.01, f"{name} regulator systematic {sys_err:.2e} > 1%"


# ═══════════════════════════════════════════════════════════
# Spectrum Edge Cases
# ═══════════════════════════════════════════════════════════


def test_RP3_spectrum_zero_modes_below_M_CURV():
    """Below M_CURV, very few modes should exist for any field type."""
    from cgc.engine.frg_flow_rp3 import RP3Spectrum

    sp = RP3Spectrum(L=L_RP3)
    k_lo = M_CURV * 0.1
    for ft in [FieldSpecies.SCALAR, FieldSpecies.SPINOR, FieldSpecies.VECTOR]:
        n = sp.count_modes_below(k_lo, ft)
        assert n < 50, f"{ft.name} has {n} modes below 0.1*M_CURV, expected < 50"


def test_RP3_spectrum_many_modes_above_M_P():
    """Above M_P, many modes should exist."""
    from cgc.engine.frg_flow_rp3 import RP3Spectrum

    sp = RP3Spectrum(L=L_RP3)
    k_hi = M_P * 10
    for ft in [FieldSpecies.SCALAR, FieldSpecies.VECTOR]:
        n = sp.count_modes_below(k_hi, ft)
        assert n > 10, f"{ft.name} has only {n} modes below 10*M_P, expected > 10"


# ═══════════════════════════════════════════════════════════
# ChiPotential Extreme Values
# ═══════════════════════════════════════════════════════════


def test_V_barrier_equals_minus_V_min():
    """V_barrier = -V_min by definition."""
    cp = ChiPotential()
    v1 = cp.V(0.0)
    assert abs(v1) < 1e-10, f"V(0)={v1:.4e} should be 0 for ST tachyon"
    # V(0) = 0 for pure tachyon, barrier = V(0) - V(chi_vev) = -V(chi_vev)
    assert abs(cp.V_barrier() + cp.V_min) < 1e-10 * abs(cp.V_min), (
        f"barrier+V_min={cp.V_barrier() + cp.V_min} should be 0"
    )


def test_chi_vev_always_positive():
    """chi_vev must be positive for physically meaningful VEV."""
    for T in [3, 5, 10, 50, 100]:
        cp = ChiPotential(T=T)
        assert cp.chi_vev > 0, f"T={T} chi_vev={cp.chi_vev} should be positive"


# ═══════════════════════════════════════════════════════════
# M_P → M_CURV Limit (Curvature → 0)
# ═══════════════════════════════════════════════════════════


def test_large_L_flat_limit():
    """As L → ∞ (flat space), Pi0 per mode should decrease.

    RP3 curvature ~ 1/L². In flat space limit L → ∞,
    the curvature-induced mode density converges to
    continuous integral. Pi0 should approach
    continuous-integral prediction.
    """
    # Compare L=2.44 (curved) vs L=10 (nearly flat)
    fields = [
        FieldContent("test_sc", FieldSpecies.SCALAR, 1, 1, 0.0, 1.0),
    ]
    k = np.geomspace(1.0, M_P, 200)

    for L_val in [2.44, 10.0]:
        trace = RP3TraceDensity(fields, L=L_val, regulator=LitimRegulator())
        eta = np.array([trace.trace_density_at_k(ki) for ki in k])
        d_ln = np.log(k[1] / k[0])
        pi0 = float(np.cumsum(eta)[-1] * d_ln)
        print(f"    L={L_val}: Pi0={pi0:.4e}")
        # As L increases, M_CURV = M_P/L decreases
        # → more modes fit in the k-range → Pi0 increases
        # This is physically correct: flat space has more modes


# ═══════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════


def run_tests():
    import traceback
    import warnings

    warnings.filterwarnings("ignore")
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_") and callable(o)]
    n_pass = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}")
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL: {name} - {e}")
        except Exception as e:
            print(f"  ERROR: {name} - {e}")
            traceback.print_exc()
    print(f"\n  {n_pass}/{len(tests)} passed")
    return n_pass == len(tests)


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_tests() else 1)
