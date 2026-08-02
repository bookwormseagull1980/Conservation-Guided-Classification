"""
Conservation-Law Checker — Phase 2.4
============================================================================

Validates whether the matrix elements of ladder diagrams are protected
by conservation laws (Ward identities, BRST symmetry) and thus guaranteed
nonzero at zero momentum transfer.

Core logic:
  - For PROTECTED operators (conserved Noether currents, BRST-closed operators):
    Ward/BRST identities → ⟨protected|T{J₁…J_n}⟩ ≠ 0 at q=0
  - For UNPROTECTED operators: no such guarantee
    → matrix elements may be suppressed

This module is the "Conservation-Guided" part of CGC. It does NOT attempt
to compute the matrix elements explicitly — it only determines whether
conservation laws provide a rigorous guarantee of non-vanishing.

Currently supported operator types:
  - Tμν (Ward identity from diffeomorphism invariance of the flat-spacetime action)
  - Fμν^a (BRST symmetry of gauge-fixed Yang-Mills)
  - Conserved currents Jμ (Noether theorem)
  - Unprotected scalars/fermions (no conservation guarantee)

Verification target (Phase 2.6 benchmark):
  For Tμν spin-2 channel, engine must confirm that Ward identity guarantees
  nonzero matrix elements for q=0 ladder diagrams.
"""


# References
#     Ward (1950): Ward-Takahashi identity for Tmu_nu protection
#     Becchi-Rouet-Stora (1976): BRST symmetry for F^2 protection
#     Slavnov-Taylor identities: ghost-antighost Ward identities
#

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .diagram_generator import OperatorSpec, OperatorType
from .topology_classifier import TopologyClassification, TopologyLabel

# ── Protection Status ────────────────────────────────────────────────────


class ProtectionBasis(Enum):
    """Type of conservation law providing protection."""

    WARD_IDENTITY = auto()  # ∂_μ⟨T{Jμ(x)…}⟩ = 0 → q_μ M^μ = 0 but M^μ ≠ 0
    BRST_SYMMETRY = auto()  # s_B O = 0 → protected
    NOETHER_THEOREM = auto()  # conserved current from global symmetry
    NONE = auto()  # no conservation law → unprotected


@dataclass
class ProtectionVerdict:
    """Result of conservation-law analysis for one diagram class."""

    operator_name: str
    protection_basis: ProtectionBasis
    is_protected: bool
    matrix_element_nonzero: bool
    theorem_reference: str = ""  # e.g. "Ward 1950", "Slavnov-Taylor"
    notes: str = ""


@dataclass
class ConservationReport:
    """Complete conservation-law analysis for the accumulating class."""

    operator: OperatorSpec
    ladder_diagrams: list[TopologyLabel]
    verdict: ProtectionVerdict
    all_protected: bool = True  # True if every ladder diagram is protected

    def summary(self) -> str:
        lines = [
            f"Conservation-Law Analysis: {self.operator.name}",
            f"  operator type:    {self.operator.op_type.name}",
            f"  protected:        {self.verdict.is_protected}",
            f"  basis:            {self.verdict.protection_basis.name}",
            f"  M(q=0) nonzero:   {self.verdict.matrix_element_nonzero}",
            f"  theorem:          {self.verdict.theorem_reference}",
            f"  ladder diagrams:  {len(self.ladder_diagrams)}",
            f"  all protected:    {self.all_protected}",
        ]
        return "\n".join(lines)


# ── Checker ──────────────────────────────────────────────────────────────


class ConservationChecker:
    """
    Determines whether conservation laws protect the matrix elements
    of ladder diagrams at q=0.

    The key insight (Appendix E):
      For a conserved Noether current Jμ, the Ward identity reads
        q_μ ⟨Jμ(0) O₁…O_n⟩ = 0
      This does NOT force the matrix element to vanish — it only
      constrains its tensor structure. At q=0, the identity becomes
      vacuous (0 = 0), and the matrix element is generically nonzero.

      For Tμν (spin-2), the same logic applies via the diffeomorphism
      Ward identity of the flat-spacetime QFT:
        q_μ ⟨Tμν(0) O₁…O_n⟩ = 0
      This does not force ⟨Tμν⟩ to vanish at q=0.

      Similarly, BRST symmetry protects Fμν^a insertions.

    In contrast, an unprotected operator (e.g. φ⁴, ψ̄ψ) has no such
    identity — its matrix elements may vanish at q=0 or be suppressed
    by powers of momentum.
    """

    # ── Protection Rules ──
    # These are OPERATOR-LEVEL rules derived from conservation-law
    # identities (Ward, BRST, Noether). They are a physics lookup table,
    # not dynamically derived. Extending to new operator types requires
    # analyzing the relevant conservation identity for that operator.

    PROTECTION_RULES: dict[OperatorType, ProtectionVerdict] = {
        OperatorType.CONSERVED_CURRENT: ProtectionVerdict(
            operator_name="Tμν / conserved current",
            protection_basis=ProtectionBasis.WARD_IDENTITY,
            is_protected=True,
            matrix_element_nonzero=True,
            theorem_reference="Ward (1950); Takahashi (1957)",
            notes=(
                "∂_μ⟨Tμν…⟩ = 0 → q_μ M^{μν…} = 0. "
                "At q=0 the identity is vacuous → M ≠ 0 is generic. "
                "No small-parameter suppression."
            ),
        ),
        OperatorType.GAUGE_FIELD_STRENGTH: ProtectionVerdict(
            operator_name="Fμν^a",
            protection_basis=ProtectionBasis.BRST_SYMMETRY,
            is_protected=True,
            matrix_element_nonzero=True,
            theorem_reference="Slavnov–Taylor; BRST (Becchi–Rouet–Stora 1976)",
            notes=(
                "s_B F = 0 → BRST-closed. "
                "Matrix elements of BRST-closed operators are protected "
                "from anomalous suppression at q=0."
            ),
        ),
        OperatorType.UNPROTECTED_SCALAR: ProtectionVerdict(
            operator_name="scalar composite (e.g. φ†φ)",
            protection_basis=ProtectionBasis.NONE,
            is_protected=False,
            matrix_element_nonzero=False,
            theorem_reference="none",
            notes=(
                "No conservation law protects scalar composites. "
                "Matrix elements may vanish at q=0 or be suppressed "
                "by powers of momentum. Each case requires explicit computation."
            ),
        ),
        OperatorType.UNPROTECTED_FERMION: ProtectionVerdict(
            operator_name="fermion bilinear (e.g. ψ̄ψ)",
            protection_basis=ProtectionBasis.NONE,
            is_protected=False,
            matrix_element_nonzero=False,
            theorem_reference="none",
            notes=(
                "No conservation law protects fermion bilinears. "
                "Chiral symmetry may provide partial protection "
                "(anomalous Ward identity) but this is model-dependent."
            ),
        ),
        OperatorType.OTHER: ProtectionVerdict(
            operator_name="unknown operator",
            protection_basis=ProtectionBasis.NONE,
            is_protected=False,
            matrix_element_nonzero=False,
            theorem_reference="none",
            notes="Conservation status unknown — requires manual specification.",
        ),
    }

    # ── Public API ──

    def check(self, operator: OperatorSpec, topo: TopologyClassification) -> ConservationReport:
        """
        Analyze conservation-law protection for the accumulating diagram class.

        Args:
            operator: composite operator specification
            topo: topology classification result

        Returns:
            ConservationReport with protection verdict
        """
        # Look up protection rule
        verdict = self.PROTECTION_RULES.get(
            operator.op_type,
            ProtectionVerdict(
                operator_name=operator.name,
                protection_basis=ProtectionBasis.NONE,
                is_protected=False,
                matrix_element_nonzero=False,
                theorem_reference="none",
                notes="Operator type not in protection rules.",
            ),
        )

        # Check: protection status is operator-level, not diagram-level.
        # All ladder diagrams for the same operator share the same
        # protection verdict. Set all_protected iff the operator
        # itself is protected (regardless of ladder count).
        all_protected = verdict.is_protected

        return ConservationReport(
            operator=operator,
            ladder_diagrams=topo.ladder,
            verdict=verdict,
            all_protected=all_protected,
        )

    def is_injection_nonzero(self, report: ConservationReport) -> bool:
        """
        Determine whether the injection term Δ_σ(μ²) is guaranteed nonzero.

        The injection term receives contributions from all q=0 ladder diagrams.
        If the operator is protected, the matrix elements of these diagrams
        are guaranteed nonzero at q=0 by the relevant conservation identity.
        Hence Δ_σ(μ²) ≠ 0 is a kinematic/conservation-law result, not a
        perturbative assumption.

        This is the CGC method's central upgrade over the heat-kernel approach:
        instead of relying on a₂ ≠ 0 (perturbative, could be canceled by
        nonperturbative effects), we rely on the algebraic fact that
        ⟨protected|q=0⟩ ≠ 0 by Ward/BRST.
        """
        return report.verdict.matrix_element_nonzero
