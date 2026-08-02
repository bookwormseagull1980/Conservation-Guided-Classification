"""
Tμν Spin-2 Channel — Benchmark Operator
============================================================================

Energy-momentum tensor, spin-2 projection.
This is the PRIMARY benchmark for the CGC engine (Phase 2.6).

The operator is a conserved Noether current (Ward identity protected).
At one-loop, the complete diagram enumeration and classification is
known from Appendix E of Paper 1 and serves as the ground truth for
engine validation.

Reference:
  - Appendix E, Figures 1–3
  - Eqs. (E.1)–(E.5)
  - sec02_spectral_macro.tex, sec04_pole_existence.tex
"""

from ..engine.diagram_generator import OperatorSpec, OperatorType


def TMunuSpin2() -> OperatorSpec:
    """
    Energy-momentum tensor, spin-2 transverse-traceless projection.

    Properties:
      - Lorentz rank: 2 (tensor)
      - Spin channel: 2
      - Mass dimension: 4
      - Protected by: Ward identity (diffeomorphism invariance of flat-spacetime action)
      - External legs: 2 (two-point function → spectral density)

    One-loop enumeration from first principles:
      Tμν couples to ALL SM fields (it's the Noether current of
      spacetime translations). For each coupled field type:
        (a) q=0 bubble — back-to-back fast modes at each insertion
            → contributes to continuum ρ_cont and Π₀(q=0)
        (b) q≠0 variant — net momentum transfer to slow modes
            → Gaussian suppression → Langevin noise kernel N(q)
      Ladder diagrams (multi-loop) are constructed by the
      resummation module from Π₀, not by the one-loop generator.

    Benchmark reference data:
      total_diagrams = 6  (3 field types × 2 kinematic classes)
      q0_count = 3        (weyl_fermion, gauge_boson, real_scalar bubbles)
      q_nonzero_count = 3  (one per field type)
      bubble_count = 3     (one per field type, each n_bubbles=1)
      ladder_count = 0     (ladders are multi-loop, built by resummation)
      protected = True
      injection_nonzero = True
    """
    return OperatorSpec(
        name="Tμν spin-2 (TT projection)",
        op_type=OperatorType.CONSERVED_CURRENT,
        lorentz_rank=2,
        spin_channel=2,
        external_momenta=2,
        mass_dimension=4,
        is_protected=True,
        protection_source="Ward identity (d_mu T^mu_nu = 0 => q_mu M^{mu nu} = 0)",
    )


# ── Benchmark Reference Data ──

TMUNU_SPIN2_BENCHMARK = {
    "expected_total_diagrams": 6,
    "expected_q0_count": 3,
    "expected_q_nonzero_count": 3,
    "expected_bubble_count": 3,
    "expected_ladder_count": 0,
    "expected_protected": True,
    "expected_injection_nonzero": True,
    "description": (
        "One-loop Tμν spin-2 channel. "
        "3 SM field types (weyl_fermion, gauge_boson, real_scalar) × "
        "2 kinematic classes (q=0 bubble + q≠0 variant) = 6 diagrams. "
        "Ladder diagrams are multi-loop structures built by the "
        "resummation module from Π₀, not by the one-loop generator. "
        "Injection term Δ_σ is guaranteed nonzero by Ward identity."
    ),
}
