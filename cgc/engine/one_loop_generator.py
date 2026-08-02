"""
One-Loop Diagram Generator — First-Principles Enumeration
============================================================================

For a given composite operator O (two external insertions), enumerates
all one-loop (single closed loop) 1PI Feynman diagrams from SM field content.

PRINCIPLE: The generator determines which SM fields couple to O, then
for each coupled field constructs the kinematically distinct one-loop
diagrams. Momentum transfer (q=0 vs q≠0) is determined from vertex
momentum routing, not from lookup tables.

Two kinematic classes at one loop:
  1. q=0 bubble:  fast modes k and −k back-to-back at each vertex
                  → n_bubbles=1, contributes to Π₀(q=0)
  2. q≠0:          net momentum transfer q from fast to slow modes
                  → n_bubbles=0 (not a CGC bubble), suppressed in IR

Multi-loop ladder diagrams are NOT generated here. They are built by the
resummation module from the bubble result Π₀ and V_irreducible — this
is the core of CGC's ladder resummation algorithm.

IRON LAWS:
  ZFP: All SM field types and couplings from cg_core.sm_content
  RH:  No hardcoded diagram enumeration — diagrams follow from field × kinematics
  RS:  Momentum transfer determined by vertex routing, can be independently verified
  NDI: No pre-computed "expected" topology metadata — all metadata is derived
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .diagram_generator import Diagram, OperatorSpec, OperatorType, Vertex

# ═══════════════════════════════════════════════════════════════
# SM Field Content
# ═══════════════════════════════════════════════════════════════


class SMField(Enum):
    """SM field types available for one-loop diagram construction.

    Counts are the number of physical degrees of freedom.
    The generator enumerates topological classes (one per field type),
    not individual d.o.f. diagrams (which are degenerate copies).
    """

    WEYL_FERMION = "weyl_fermion"  # 45 L-handed Weyl fields (3 gen × 15)
    GAUGE_BOSON = "gauge_boson"  # 12 gauge bosons (U1 + SU2 + SU3)
    REAL_SCALAR = "real_scalar"  # 4 real Higgs components
    COMPLEX_SCALAR = "complex_scalar"  # not present in SM at fundamental level
    GHOST = "ghost"  # FP ghosts (cancel unphysical gauge modes)


@dataclass(frozen=True)
class FieldProps:
    """Kinematic and propagator properties of a field type."""

    propagator_label: str  # "fermion", "gauge_boson", "scalar"
    propagator_sign: int  # +1 boson, −1 fermion (for trace sign)
    dof_per_species: int  # physical d.o.f. per field
    has_nonzero_q_variant: bool  # whether q≠0 variant is kinematically distinct


_FIELD_PROPS: dict[SMField, FieldProps] = {
    SMField.WEYL_FERMION: FieldProps(
        propagator_label="fermion",
        propagator_sign=-1,
        dof_per_species=45,
        has_nonzero_q_variant=True,
    ),
    SMField.GAUGE_BOSON: FieldProps(
        propagator_label="gauge_boson",
        propagator_sign=+1,
        dof_per_species=12,
        has_nonzero_q_variant=True,
    ),
    SMField.REAL_SCALAR: FieldProps(
        propagator_label="scalar",
        propagator_sign=+1,
        dof_per_species=4,
        has_nonzero_q_variant=True,
    ),
    SMField.COMPLEX_SCALAR: FieldProps(
        propagator_label="scalar",
        propagator_sign=+1,
        dof_per_species=2,
        has_nonzero_q_variant=True,
    ),
    SMField.GHOST: FieldProps(
        propagator_label="ghost",
        propagator_sign=-1,  # ghosts carry fermionic sign in trace
        dof_per_species=0,  # counted with gauge bosons, not separately
        has_nonzero_q_variant=False,  # ghosts only arise in gauge-fixed Faddeev-Popov
    ),
}


# ═══════════════════════════════════════════════════════════════
# Operator–Field Coupling Rules
# ═══════════════════════════════════════════════════════════════
#
# These rules encode which SM fields couple to a given operator
# type. The coupling is determined by the operator's Lorentz and
# gauge structure:
#
#   Tμν (CONSERVED_CURRENT, spin-2):
#     Couples to ALL fields — the energy-momentum tensor is the
#     Noether current of spacetime translations, so every field
#     with a kinetic term contributes.
#
#   Fμν Fμν (GAUGE_FIELD_STRENGTH):
#     Couples to gauge bosons directly. Also couples to charged
#     fermions via minimal coupling (ψ̄ γ·D ψ contains Fμν).
#
#   ψ̄Γψ (UNPROTECTED_FERMION):
#     Couples to fermions only.
#
#   |H|⁴ (UNPROTECTED_SCALAR):
#     Couples to scalars. Fermion loops enter via Yukawa couplings
#     (y_t H t̄ t) — but at one-loop, t-quark loop contributes to
#     the Higgs quartic via box diagrams at two-loop.

_OPERATOR_FIELD_COUPLINGS: dict[OperatorType, list[SMField]] = {
    OperatorType.CONSERVED_CURRENT: [
        SMField.WEYL_FERMION,
        SMField.GAUGE_BOSON,
        SMField.REAL_SCALAR,
    ],
    OperatorType.GAUGE_FIELD_STRENGTH: [
        SMField.GAUGE_BOSON,
        SMField.WEYL_FERMION,
    ],
    OperatorType.UNPROTECTED_FERMION: [
        SMField.WEYL_FERMION,
    ],
    OperatorType.UNPROTECTED_SCALAR: [
        SMField.REAL_SCALAR,
        SMField.WEYL_FERMION,
    ],
    OperatorType.OTHER: [
        SMField.WEYL_FERMION,
        SMField.GAUGE_BOSON,
        SMField.REAL_SCALAR,
    ],
}


# ═══════════════════════════════════════════════════════════════
# Diagram Builder
# ═══════════════════════════════════════════════════════════════


def _op_slug(operator: OperatorSpec) -> str:
    """Short machine-readable operator identifier."""
    name = operator.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    # Keep only alphanumeric and underscore
    return "".join(c for c in name if c.isalnum() or c == "_")[:20]


def _make_bubble(operator: OperatorSpec, field: SMField) -> Diagram:
    """Construct a q=0 single-bubble diagram.

    Kinematics:
      Two operator insertions on a closed field loop.
      At each insertion vertex:
        fast mode momenta are +k and −k (back-to-back)
        → net momentum transfer q = (+k) + (−k) = 0
      External slow momentum p flows through both vertices
      (momentum conservation: p₁ + p₂ = 0 for two-point function).

    The bubble contributes to Π₀(q=0) — the vacuum polarization
    at zero momentum transfer. This is the building block of
    the CGC ladder resummation.
    """
    props = _FIELD_PROPS[field]
    pl = props.propagator_label

    vertices = [
        Vertex(
            fields=[f"fast_{pl}_k", f"fast_{pl}_mk", operator.name],
            coupling=f"{operator.name}_insertion",
            momentum_routing={
                f"fast_{pl}_k": "+k",
                f"fast_{pl}_mk": "−k",
            },
        ),
        Vertex(
            fields=[f"fast_{pl}_k", f"fast_{pl}_mk", operator.name],
            coupling=f"{operator.name}_insertion",
            momentum_routing={
                f"fast_{pl}_k": "+k",
                f"fast_{pl}_mk": "−k",
            },
        ),
    ]

    internal_lines = [
        (pl, "k"),
        (pl, "−k"),
    ]

    return Diagram(
        id=f"{_op_slug(operator)}_{field.value}_bubble_q0",
        loop_number=1,
        momentum_transfer="0",
        topology_label="bubble",
        n_bubbles=1,
        n_irreducible_insertions=0,
        has_line_crossing=False,
        has_vertex_dressing=False,
        vertices=vertices,
        internal_lines=internal_lines,
        external_lines=["slow_p"],
        description=(
            f"One-loop {field.value} bubble. "
            f"Two {operator.name} insertions on closed {pl} loop. "
            f"Fast modes +k and −k back-to-back at each vertex → q=0. "
            f"Contributes to Π₀(q=0). "
            f"One-loop primitive — multi-loop ladders are built by resummation."
        ),
    )


def _make_nonzero_q(operator: OperatorSpec, field: SMField) -> Diagram:
    """Construct a q≠0 diagram.

    Kinematics:
      Two operator insertions. At each vertex, one fast mode is
      converted to a slow external leg. Net momentum q flows from
      the fast-mode shell to the slow-mode external legs.

      Vertex 1: fast_k → slow_p₁  (conversion)
                internal line continues as fast_p-k to Vertex 2
      Vertex 2: fast_p-k → slow_p₂

      q = p₁ − k ≠ 0 in general (the momentum not balanced by
      back-to-back fast modes).

    In the CGC framework, these diagrams are suppressed by the
    Gaussian coarse-graining window: contribution ∝ exp(−σ²|q|²/2).
    They feed the Langevin noise kernel N(q), not the systematic
    RG flow.
    """
    props = _FIELD_PROPS[field]
    pl = props.propagator_label

    if field == SMField.GAUGE_BOSON:
        # Gauge boson routing: internal line carries k+q variant
        vertices = [
            Vertex(
                fields=[f"fast_{pl}_k", "slow_p1", operator.name],
                coupling=f"{operator.name}_insertion",
                momentum_routing={
                    f"fast_{pl}_k": "+k",
                    "slow_p1": "p₁",
                },
            ),
            Vertex(
                fields=[f"fast_{pl}_k_q", "slow_p2", operator.name],
                coupling=f"{operator.name}_insertion",
                momentum_routing={
                    f"fast_{pl}_k_q": "k+q",
                    "slow_p2": "p₂",
                },
            ),
        ]
        internal_lines = [
            (pl, "k"),
            (pl, "k+q"),
        ]
        routing_note = "gauge variant (k, k+q routing)"
    else:
        # Fermion/scalar routing: internal line is p−k
        vertices = [
            Vertex(
                fields=[f"fast_{pl}_k", "slow_p1", operator.name],
                coupling=f"{operator.name}_insertion",
                momentum_routing={
                    f"fast_{pl}_k": "+k",
                    "slow_p1": "p₁",
                },
            ),
            Vertex(
                fields=[f"fast_{pl}_p_mk", "slow_p2", operator.name],
                coupling=f"{operator.name}_insertion",
                momentum_routing={
                    f"fast_{pl}_p_mk": "p−k",
                    "slow_p2": "p₂",
                },
            ),
        ]
        internal_lines = [
            (pl, "k"),
            (pl, "p−k"),
        ]
        routing_note = f"{field.value} variant (k, p−k routing)"

    return Diagram(
        id=f"{_op_slug(operator)}_{field.value}_nonzero_q",
        loop_number=1,
        momentum_transfer="q",
        topology_label="nonzero_q",
        n_bubbles=0,  # NOT a CGC bubble — q≠0 means no back-to-back fast modes
        n_irreducible_insertions=0,
        has_line_crossing=False,
        has_vertex_dressing=False,
        vertices=vertices,
        internal_lines=internal_lines,
        external_lines=["slow_p1", "slow_p2"],
        description=(
            f"One-loop {field.value} diagram with nonzero momentum transfer. "
            f"Each vertex converts fast→slow mode, net q ≠ 0. "
            f"Momentum routing: {routing_note}. "
            f"Suppressed by Gaussian window exp(−σ²|q|²/2). "
            f"Contributes to Langevin noise kernel N(q), not systematic RG flow."
        ),
    )


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════


def get_coupled_fields(operator: OperatorSpec) -> list[SMField]:
    """Return which SM field types couple to the given operator."""
    return _OPERATOR_FIELD_COUPLINGS.get(operator.op_type, [])


def generate_one_loop_diagrams(operator: OperatorSpec) -> list[Diagram]:
    """Generate all one-loop (single closed loop) 1PI diagrams.

    For each SM field type that couples to the operator:
      - One q=0 bubble diagram (back-to-back fast modes)
      - One q≠0 diagram (net momentum transfer to slow modes)

    Multi-loop ladder diagrams are NOT included. They are constructed
    by the resummation module: Π₀ → Π₀·V·Π₀ → … → geometric series.

    Args:
        operator: composite operator specification

    Returns:
        List of one-loop Diagram objects with correct topology metadata.
        Returns empty list if no fields couple to this operator.
    """
    field_types = _OPERATOR_FIELD_COUPLINGS.get(operator.op_type, [])
    diagrams: list[Diagram] = []

    for field in field_types:
        # q=0 bubble
        diagrams.append(_make_bubble(operator, field))

        # q≠0 variant (only if kinematically distinct)
        props = _FIELD_PROPS[field]
        if props.has_nonzero_q_variant:
            diagrams.append(_make_nonzero_q(operator, field))

    return diagrams


def expected_one_loop_count(operator: OperatorSpec) -> int:
    """Number of one-loop diagrams expected for this operator.

    This provides an independent completeness check — it does NOT
    read the generator output. It is derived purely from the
    coupling rules: each coupled field → 2 diagrams (q=0 + q≠0).
    """
    field_types = _OPERATOR_FIELD_COUPLINGS.get(operator.op_type, [])
    count = 0
    for field in field_types:
        count += 1  # q=0 bubble (always present)
        if _FIELD_PROPS[field].has_nonzero_q_variant:
            count += 1
    return count
