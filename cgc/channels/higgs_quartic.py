"""
Higgs Quartic Coupling Channel — Phase 3 Application
============================================================================

λ(φ†φ)² operator. Scalar, Lorentz rank 0.
Unprotected operator — no conservation law, no Ward identity.

The CGC engine predicts:
  - Zero momentum transfer ladder diagrams exist
  - BUT matrix elements are NOT protected
  - Injection term Δ_σ is expected to be power-suppressed
  - No δ-pole formation under coarse-graining

This prediction is directly relevant to Paper 2: the Higgs quartic
coupling's ultraviolet boundary value is NOT determined by coarse-graining
dynamics. Zero is the most natural choice (λ(M_P) = 0, matching the
paper's Case 4 discussion).

If CGC confirms: no accumulation of zero-modes in the Higgs quartic channel
→ independent theoretical support for Paper 2's boundary condition.

Reference:
  - Phase 3 development plan, section 4.1
  - Paper 2 (arxiv-jhep): Higgs quartic coupling boundary condition
"""

from ..engine.diagram_generator import OperatorSpec, OperatorType

HIGGS_QUARTIC_BENCHMARK = {
    "expected_total_diagrams": 4,
    "expected_q0_count": 2,
    "expected_q_nonzero_count": 2,
    "expected_bubble_count": 2,
    "expected_ladder_count": 0,
    "expected_protected": False,
    "expected_injection_nonzero": False,
    "description": (
        "One-loop Higgs quartic λ(φ†φ)² channel. "
        "2 SM field types (real_scalar, weyl_fermion via Yukawa) × "
        "2 kinematic classes (q=0 bubble + q≠0 variant) = 4 diagrams. "
        "Unprotected — no Ward/BRST identity for scalar composites. "
        "Injection term Δ_σ → 0. "
        "Supports Paper 2 boundary condition: λ(M_G) ≈ 0 is natural."
    ),
}


def HiggsQuartic() -> OperatorSpec:
    """
    Higgs quartic coupling λ(φ†φ)² operator.

    Properties:
      - Lorentz rank: 0 (scalar)
      - Spin channel: 0
      - Mass dimension: 4
      - Protected by: NONE
      - External legs: 4 (four-point function)

    One-loop diagrams:
      - Single bubble: scalar loop with quartic insertion → continuum
      - Ladder: multiple bubbles with scalar exchanges → exists topologically
      - BUT: no Ward/BRST protection → matrix elements suppressed at q=0

    CGC prediction:
      injection_nonzero = False → no pole accumulation
    """
    return OperatorSpec(
        name="Higgs quartic λ(φ†φ)²",
        op_type=OperatorType.UNPROTECTED_SCALAR,
        lorentz_rank=0,
        spin_channel=0,
        external_momenta=4,
        mass_dimension=4,
        is_protected=False,
        protection_source="None — scalar composites have no conservation-law protection",
    )
