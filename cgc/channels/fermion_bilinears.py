"""
Fermion Bilinears Channel — Phase 3 Application
============================================================================

ψ̄ψ, ψ̄γ5ψ, ψ̄γμψ, etc.
Unprotected operators — no conservation law guarantees nonzero matrix elements.

The CGC engine predicts:
  - Zero momentum transfer ladder diagrams exist
  - BUT matrix elements are NOT protected → may be suppressed
  - Injection term Δ_σ may vanish or be power-suppressed in μ²/Λ²
  - No δ-pole formation expected

This is a NEGATIVE prediction: unprotected operators do NOT develop
spectral poles under coarse-graining. This serves as a null test for
the CGC method — if an unprotected operator DOES develop a pole,
the method's protection criterion is falsified.

Reference:
  - Phase 3 development plan, section 4.1
"""

from ..engine.diagram_generator import OperatorSpec, OperatorType

FERMION_BENCHMARK = {
    "expected_total_diagrams": 2,
    "expected_q0_count": 1,
    "expected_q_nonzero_count": 1,
    "expected_bubble_count": 1,
    "expected_ladder_count": 0,
    "expected_protected": False,
    "expected_injection_nonzero": False,
    "description": (
        "One-loop fermion bilinear ψ̄Γψ channel. "
        "1 SM field type (weyl_fermion) × "
        "2 kinematic classes (q=0 bubble + q≠0 variant) = 2 diagrams. "
        "Unprotected — no Ward/BRST identity. "
        "Matrix elements suppressed at q=0. "
        "Injection term Δ_σ → 0 (power-suppressed in μ²/Λ²)."
    ),
}


def FermionBilinears(bilinear_type: str = "scalar") -> OperatorSpec:
    """
    Fermion bilinear operator.

    Args:
        bilinear_type: "scalar" (ψ̄ψ), "pseudoscalar" (ψ̄γ5ψ),
                       "vector" (ψ̄γμψ), "axial" (ψ̄γμγ5ψ)

    Properties:
      - Lorentz rank: depends on type (0 for scalar, 1 for vector)
      - Mass dimension: 3
      - Protected by: NONE (no Ward identity for generic bilinears)
        Exception: conserved vector current ψ̄γμψ has a Noether protection
        if the fermion has a global U(1) symmetry — handled separately

    Note: The axial current has an anomalous Ward identity (Adler-Bell-Jackiw).
    This provides PARTIAL protection but the anomaly breaks conservation
    at the quantum level.
    """
    rank_map = {
        "scalar": 0,
        "pseudoscalar": 0,
        "vector": 1,
        "axial": 1,
    }
    is_vector = bilinear_type in ("vector", "axial")

    return OperatorSpec(
        name=f"fermion bilinear ψ̄Γψ ({bilinear_type})",
        op_type=OperatorType.UNPROTECTED_FERMION,
        lorentz_rank=rank_map.get(bilinear_type, 0),
        spin_channel=1 if is_vector else 0,
        external_momenta=2,
        mass_dimension=3,
        is_protected=False,
        protection_source="None" if not is_vector else "Noether (if global U(1)) — but anomalous for axial",
    )
