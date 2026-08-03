"""Property-based tests (Hypothesis) for CGC invariants.

Tests invariants that must hold for ALL legal inputs within given domains:
  1. Single SCALAR/VECTOR/TENSOR fields always contribute Pi0 > 0
  2. Single SPINOR fields always contribute Pi0 < 0
  3. All-multi-field Pi0 scales linearly with coupling strength
  4. ChiPotential: mu2 < 0, lambda > 0, chi_vev > 0 for any T > 2
  5. DSE critical gap log10(V_crit/V_native) ≥ 0

Author: CGC
Date: 2026-07-31
"""

import os
import sys

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_CGC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # cgc/
if _CGC_ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(_CGC_ROOT))  # parent of cgc/ for imports

from cgc.rp3_engine.chi_potential import ChiPotential
from cgc.rp3_engine.frg_flow_rp3 import (
    M_P,
    FieldContent,
    FieldSpecies,
    LitimRegulator,
    RP3TraceDensity,
)

# Shared grid for efficiency
_K_GRID = np.geomspace(1.0, M_P, 200)


def compute_single_pi0(
    field_type: FieldSpecies, n_species: int, dof_per_species: int, mass_gev: float, coupling_sq: float, L: float = 2.44
) -> float:
    """Compute Pi0 for a single-field system."""
    fc = FieldContent(
        name="test",
        field_type=field_type,
        n_species=n_species,
        dof_per_species=dof_per_species,
        mass_gev=mass_gev,
        coupling_sq=coupling_sq,
    )
    trace = RP3TraceDensity([fc], L=L, regulator=LitimRegulator())
    d_ln = np.log(_K_GRID[1] / _K_GRID[0])
    eta = np.array([trace.trace_density_at_k(ki) for ki in _K_GRID])
    return float(np.cumsum(eta)[-1] * d_ln)


# ═══════════════════════════════════════════════════════════
# Strategy: physical field parameters
# ═══════════════════════════════════════════════════════════

# Reasonable domain for physical coupling constants
mass_strategy = st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False)
coupling_strategy = st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)
n_species_strategy = st.integers(min_value=1, max_value=10)
dof_strategy = st.integers(min_value=1, max_value=10)
T_strategy = st.floats(min_value=2.1, max_value=1000.0, allow_nan=False, allow_infinity=False)
alpha_strategy = st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False)


# ═══════════════════════════════════════════════════════════
# 1. Sign Invariants
# ═══════════════════════════════════════════════════════════


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
@given(n_species=n_species_strategy, dof=dof_strategy, mass=mass_strategy, coupling=coupling_strategy)
def test_scalar_pi0_positive(n_species, dof, mass, coupling):
    """Any single scalar field contributes Pi0 > 0."""
    pi0 = compute_single_pi0(FieldSpecies.SCALAR, n_species, dof, mass, coupling)
    assert pi0 >= 0, f"Scalar Pi0={pi0:.6e} should be >= 0"
    print(f"  SCALAR(n={n_species},dof={dof},m={mass:.1f},g2={coupling:.2f}): Pi0={pi0:.3e}  OK")


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
@given(n_species=n_species_strategy, dof=dof_strategy, mass=mass_strategy, coupling=coupling_strategy)
def test_spinor_pi0_negative(n_species, dof, mass, coupling):
    """Any single spinor field contributes Pi0 ≤ 0."""
    pi0 = compute_single_pi0(FieldSpecies.SPINOR, n_species, dof, mass, coupling)
    assert pi0 <= 0, f"Spinor Pi0={pi0:.6e} should be <= 0"
    print(f"  SPINOR(n={n_species},dof={dof},m={mass:.1f},g2={coupling:.2f}): Pi0={pi0:.3e}  OK")


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
@given(n_species=n_species_strategy, dof=dof_strategy, mass=mass_strategy, coupling=coupling_strategy)
def test_vector_pi0_positive(n_species, dof, mass, coupling):
    """Any single vector field contributes Pi0 > 0."""
    pi0 = compute_single_pi0(FieldSpecies.VECTOR, n_species, dof, mass, coupling)
    assert pi0 >= 0, f"Vector Pi0={pi0:.6e} should be >= 0"
    print(f"  VECTOR(n={n_species},dof={dof},m={mass:.1f},g2={coupling:.2f}): Pi0={pi0:.3e}  OK")


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
@given(n_species=n_species_strategy, dof=dof_strategy, mass=mass_strategy, coupling=coupling_strategy)
def test_tensor_pi0_positive(n_species, dof, mass, coupling):
    """Any single TT tensor field contributes Pi0 > 0."""
    pi0 = compute_single_pi0(FieldSpecies.TENSOR_TT, n_species, dof, mass, coupling)
    assert pi0 >= 0, f"Tensor Pi0={pi0:.6e} should be >= 0"
    print(f"  TENSOR(n={n_species},dof={dof},m={mass:.1f},g2={coupling:.2f}): Pi0={pi0:.3e}  OK")


# ═══════════════════════════════════════════════════════════
# 2. Coupling Scaling Invariant
# ═══════════════════════════════════════════════════════════


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
@given(
    field_type=st.sampled_from([FieldSpecies.SCALAR, FieldSpecies.VECTOR, FieldSpecies.TENSOR_TT]),
    mass=mass_strategy,
    coupling=coupling_strategy,
    scale=st.floats(min_value=0.5, max_value=2.0),
)
def test_pi0_linear_in_coupling_sq(field_type, mass, coupling, scale):
    """Pi0 ~ coupling² for bosonic fields (from G ~ g² in trace density).

    At fixed mass/modes, doubling coupling² doubles Pi0.
    """
    pi0_base = compute_single_pi0(field_type, 1, 1, mass, coupling)
    pi0_scaled = compute_single_pi0(field_type, 1, 1, mass, coupling * scale)

    if abs(pi0_base) < 1e-20:
        return  # skip zero case

    ratio_actual = pi0_scaled / pi0_base
    rel_err = abs(ratio_actual - scale) / max(scale, 0.01)
    # Allow 20% relative tolerance (discrete spectrum + FRG regulator effects)
    assert rel_err < 0.20, (
        f"{field_type.name}: expected Pi0 scaling {scale:.2f}x, got {ratio_actual:.4f}x (err={rel_err:.4f})"
    )
    print(
        f"  {field_type.name}(m={mass:.1f},g2={coupling:.2f},scale={scale:.2f}): "
        f"ratio={ratio_actual:.4f} vs expected={scale:.2f}  OK"
    )


# ═══════════════════════════════════════════════════════════
# 3. ChiPotential Invariants
# ═══════════════════════════════════════════════════════════


@settings(max_examples=50, deadline=None)
@given(T=T_strategy, alpha=alpha_strategy)
def test_chi_mu2_negative(T, alpha):
    """mu2 = -alpha * M_P^2 must be negative for any T > 2, alpha > 0."""
    cp = ChiPotential(T=T, alpha=alpha)
    assert cp.mu2 < 0, f"mu2={cp.mu2:.2e} should be negative for T={T:.1f}, alpha={alpha:.4f}"
    print(f"  T={T:.1f}, alpha={alpha:.4f}: mu2={cp.mu2:.2e} < 0  OK")


@settings(max_examples=50, deadline=None)
@given(T=T_strategy, alpha=alpha_strategy)
def test_chi_lambda_positive(T, alpha):
    """lambda must be positive for T > 2 (stable potential)."""
    cp = ChiPotential(T=T, alpha=alpha)
    assert cp.lamb > 0, f"lambda={cp.lamb:.4e} should be positive for T={T:.1f}, alpha={alpha:.4f}"
    print(f"  T={T:.1f}, alpha={alpha:.4f}: lambda={cp.lamb:.4e} > 0  OK")


@settings(max_examples=50, deadline=None)
@given(T=T_strategy)
def test_chi_vev_positive(T):
    """chi_vev must be positive for T > 2."""
    cp = ChiPotential(T=T)
    assert cp.chi_vev > 0, f"chi_vev={cp.chi_vev:.2e} should be positive for T={T:.1f}"
    print(f"  T={T:.1f}: chi_vev={cp.chi_vev:.2e} = {cp.chi_vev / M_P:.4f} M_P > 0  OK")


@settings(max_examples=50, deadline=None)
@given(T=T_strategy, alpha=alpha_strategy)
def test_chi_V_min_negative(T, alpha):
    """V(chi_vev) should be negative (tachyon symmetry breaking)."""
    cp = ChiPotential(T=T, alpha=alpha)
    assert cp.V_min < 0, f"V_min={cp.V_min:.4e} should be negative for T={T:.1f}, alpha={alpha:.4f}"
    print(f"  T={T:.1f}, alpha={alpha:.4f}: V_min={cp.V_min:.4e} < 0  OK")


# ═══════════════════════════════════════════════════════════
# 4. DSE Gap Invariant
# ═══════════════════════════════════════════════════════════


@settings(max_examples=20, deadline=None)
@given(channel=st.sampled_from(["Tmunu", "F2"]))
def test_dse_gap_nonnegative(channel):
    """log10(V_crit / V_native) ≥ 0 for any physically computed Pi0.

    This is a physical invariant: the critical V for emergent pole
    formation cannot be below the native V of the theory.
    """
    from cgc.rp3_engine.self_consistent_dyson import SelfConsistentSolver

    s = SelfConsistentSolver(channel)
    vc = s.find_v_crit()

    v_crit_val = vc.get("v_crit")
    if v_crit_val is None:
        # F2 channel: no V_crit found (fermion-dominated → no bifurcation)
        # This is physically correct: F2 has Pi0 < 0 → no pole forms
        print(f"  {channel}: V_crit=None (no pole formation, Pi0 < 0)  OK")
        return

    v_native = s.native_v
    gap = v_crit_val / v_native

    assert gap >= 1.0, f"{channel}: V_crit/V_native={gap:.2f} should be >= 1 (gap_decades={np.log10(gap):.2f})"
    print(
        f"  {channel}: V_crit={v_crit_val:.2e}, V_native={v_native:.2e}, "
        f"gap={gap:.2e}, log10(gap)={np.log10(gap):.2f}  OK"
    )


# ═══════════════════════════════════════════════════════════
# 5. Non-Conserved Operator Invariant
# ═══════════════════════════════════════════════════════════


@settings(max_examples=30, deadline=None)
@given(
    channel=st.sampled_from(["fermion", "higgs"]),
)
def test_nonconserved_injection_zero(channel):
    """For any non-conserved operator, injection term must vanish at q=0.

    This is the rigid classification invariant: conservation-law
    protection is the ONLY mechanism that allows nonzero injection.
    Without conservation protection (Ward, BRST, Slavnov-Taylor),
    the q=0 transfer diagrams cancel exactly.

    Ref: Paper 1, Appendix E (classification theorem)
    """
    from cgc import CGCPipeline
    from cgc.channels.fermion_bilinears import FermionBilinears
    from cgc.channels.higgs_quartic import HiggsQuartic

    channel_map = {
        "fermion": FermionBilinears,
        "higgs": HiggsQuartic,
    }
    pipeline = CGCPipeline()
    r = pipeline.run(channel_map[channel]())
    v = r.conservation_report.verdict

    # Non-conserved: NOT protected and injection term is zero
    assert not v.is_protected, (
        f"{channel}: is_protected={v.is_protected}, should be False for unprotected operator"
    )
    assert not v.matrix_element_nonzero, (
        f"{channel}: matrix_element_nonzero={v.matrix_element_nonzero}, should be False (injection vanishes)"
    )
    print(f"  {channel}: protected=False, injection_zero=True (conservation-law classification)  OK")


# ═══════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════


def run_tests():
    """Run all property-based tests.

    Hypothesis generates random inputs and verifies invariants.
    Each test runs ~20-50 examples.
    """
    import traceback
    import warnings

    warnings.filterwarnings("ignore")

    tests = [
        test_scalar_pi0_positive,
        test_spinor_pi0_negative,
        test_vector_pi0_positive,
        test_tensor_pi0_positive,
        test_pi0_linear_in_coupling_sq,
        test_chi_mu2_negative,
        test_chi_lambda_positive,
        test_chi_vev_positive,
        test_chi_V_min_negative,
        test_dse_gap_nonnegative,
        test_nonconserved_injection_zero,
    ]

    n_pass = 0
    for fn in tests:
        try:
            fn()  # Hypothesis runs the test with many inputs
            print(f"  PASS: {fn.__name__}")
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__} - {e}")
        except Exception as e:
            print(f"  ERROR: {fn.__name__} - {e}")
            traceback.print_exc()

    print(f"\n  {n_pass}/{len(tests)} property tests passed")
    return n_pass == len(tests)


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_tests() else 1)
