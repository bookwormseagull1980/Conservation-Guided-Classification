"""
Gauge Field Strength Channel — Phase 2 Application
============================================================================

Fμν^a operator, spin-1 projection.
First independent application of the CGC engine beyond the benchmark.

The operator is BRST-protected (s_B F = 0 for gauge-invariant Fμν^a F^{aμν}).
The logic is parallel to Tμν:
  - Zero momentum transfer ladder diagrams have matrix elements
    protected by BRST symmetry
  - Nonzero momentum transfer diagrams are suppressed by oscillatory factors
  - Ladder resummation → geometric series → potential spectral pole formation

This channel is the primary target for FRG-CGC cross-validation (Phase 3.3):
  - Does the gauge field strength spectral function develop a δ-pole?
  - If yes: compare k_crit(CGC) vs k_crit(FRG)
  - Quantify ladder approximation accuracy

Reference:
  - Phase 2 development plan, sections 3.1–3.3
"""

from ..engine.diagram_generator import OperatorSpec, OperatorType


def GaugeFieldStrength(gauge_group: str = "SU(3)") -> OperatorSpec:
    """
    Gauge field strength Fμν^a, spin-1 channel.

    Properties:
      - Lorentz rank: 2 (tensor) but spin-1 gauge field
      - Spin channel: 1
      - Mass dimension: 4
      - Protected by: BRST symmetry (s_B F = 0 for Tr[F∧*F])
      - External legs: 2

    For SU(N) gauge theory, the one-loop diagrams are:
      - Single bubble: gluon loop with Fμν insertion → continuum
      - Ladder: multiple bubbles with irreducible gluon exchanges → accumulating
      - Crossed ladder: non-planar gluon exchanges → to analyze

    The CGC engine should produce:
      - q=0 ladder diagrams with nonzero matrix elements (BRST guarantee)
      - Nonzero injection term Δ_σ for the gauge field strength
      - Critical condition 1 − V·Π₀(0) = 0
    """
    return OperatorSpec(
        name=f"Fμν^a ({gauge_group}) spin-1",
        op_type=OperatorType.GAUGE_FIELD_STRENGTH,
        lorentz_rank=2,
        spin_channel=1,
        external_momenta=2,
        mass_dimension=4,
        is_protected=True,
        protection_source="BRST symmetry (s_B F = 0 for gauge-invariant Tr[F∧*F])",
    )


# ── Benchmark Reference Data ──

GAUGE_FIELD_BENCHMARK = {
    "expected_total_diagrams": 4,
    "expected_q0_count": 2,
    "expected_q_nonzero_count": 2,
    "expected_bubble_count": 2,
    "expected_ladder_count": 0,
    "expected_protected": True,
    "expected_injection_nonzero": True,
    "description": (
        "One-loop Fμν^a F^{aμν} gauge field strength channel. "
        "2 SM field types (gauge_boson, weyl_fermion) × "
        "2 kinematic classes (q=0 bubble + q≠0 variant) = 4 diagrams. "
        "BRST-protected → ladder matrix elements nonzero at q=0. "
        "Gauge self-coupling bubble + quark bubble. "
        "Scalars decoupled at one-loop (Higgs is color-neutral)."
    ),
}
