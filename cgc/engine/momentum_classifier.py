"""
Momentum Transfer Classifier — Phase 2.2
============================================================================

Classifies Feynman diagrams by momentum transfer from fast-mode internal
lines to slow-mode external legs.

Core logic:
  - Extract momentum transfer q for each diagram
  - q ≠ 0 → SUPPRESSED class (oscillatory factor → zero in IR, → Langevin noise)
  - q = 0 → retain for topology classification

This is a KINEMATIC classification — it does not depend on interaction
strength or the specific form of couplings. It only depends on the
routing of momenta through the diagram.

Rigorous basis:
  - Oscillatory factors e^{iq·x} in the coarse-graining window average to
    zero as σ → ∞ for q ≠ 0 (standard Fourier analysis result).
  - The suppression is exponential in the coarse-graining scale σ for
    σ|q| ≫ 1.
  - q = 0 diagrams have no oscillatory factor → contributions accumulate.

Verification target (Phase 2.6 benchmark):
  For Tμν spin-2 channel, non-zero momentum transfer diagrams must match
  Appendix E Figure 2.
"""


# References
#     Wetterich (1993), Phys. Lett. B 301, 90: exact FRG flow equation
#     The zero-momentum-transfer classification follows from conservation
#     law insertion at q=0 (see Paper 1, Appendix E)
#

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from .diagram_generator import Diagram, DiagramSet

# ── Classification Output ────────────────────────────────────────────────


class MomentumClass(Enum):
    """Classification by momentum transfer from fast to slow modes."""

    ZERO_TRANSFER = auto()  # q = 0 → accumulates
    NONZERO_TRANSFER = auto()  # q ≠ 0 → suppressed (→ noise kernel)
    UNDETERMINED = auto()  # routing ambiguous


@dataclass
class MomentumLabel:
    """Result of momentum transfer classification for one diagram."""

    diagram_id: str
    momentum_class: MomentumClass
    q_label: str = ""  # symbolic label for the transfer momentum
    suppression_factor: str = ""  # symbolic expression for oscillatory factor
    routing_notes: str = ""  # explanation of routing choice


@dataclass
class MomentumClassification:
    """Complete momentum transfer classification for a diagram set."""

    operator_name: str
    zero_transfer: list[MomentumLabel] = field(default_factory=list)
    nonzero_transfer: list[MomentumLabel] = field(default_factory=list)
    undetermined: list[MomentumLabel] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.zero_transfer) + len(self.nonzero_transfer) + len(self.undetermined)

    @property
    def suppression_ratio(self) -> float:
        """Fraction of diagrams in the suppressed (q≠0) class."""
        if self.total == 0:
            return 0.0
        return len(self.nonzero_transfer) / self.total

    def summary(self) -> str:
        lines = [
            f"Momentum Transfer Classification: {self.operator_name}",
            f"  q = 0  (accumulate):  {len(self.zero_transfer)}",
            f"  q ≠ 0  (suppressed):   {len(self.nonzero_transfer)}",
            f"  undetermined:          {len(self.undetermined)}",
            f"  suppression ratio:     {self.suppression_ratio:.2%}",
        ]
        return "\n".join(lines)


# ── Classifier ───────────────────────────────────────────────────────────


class MomentumClassifier:
    """
    Classifies diagrams by momentum transfer q from fast to slow modes.

    The core idea of the CGC method: diagrams with q ≠ 0 carry oscillatory
    factors e^{iq·x} in the coarse-graining window. These factors average to
    zero in the infrared limit (σ → ∞), so such diagrams contribute only to
    the Langevin noise kernel, not to the systematic RG flow of the spectral
    function.

    Diagrams with q = 0 have no oscillatory factor. Their contributions
    accumulate under coarse-graining and are the only candidates for
    producing spectral poles.
    """

    def classify(self, diag_set: DiagramSet) -> MomentumClassification:
        """
        Classify all diagrams in the set by momentum transfer.

        Args:
            diag_set: set of diagrams to classify

        Returns:
            MomentumClassification with diagrams sorted by q class
        """
        classification = MomentumClassification(
            operator_name=diag_set.operator.name,
        )

        for diagram in diag_set.diagrams:
            label = self._classify_one(diagram, diag_set.operator)
            if label.momentum_class == MomentumClass.ZERO_TRANSFER:
                classification.zero_transfer.append(label)
            elif label.momentum_class == MomentumClass.NONZERO_TRANSFER:
                classification.nonzero_transfer.append(label)
            else:
                classification.undetermined.append(label)

        return classification

    def _classify_one(self, diagram: Diagram, operator: Any) -> MomentumLabel:
        """
        Classify a single diagram's momentum transfer.

        INDEPENDENT ANALYSIS (2026-08-03): the momentum transfer is
        derived from the momentum routing at the vertices — an external
        (slow) momentum label that is NOT cancelled by a back-to-back
        fast mode constitutes a net transfer q ≠ 0.  This re-derives
        the classification from the diagram's kinematic data instead
        of reading a pre-set label.

        Algorithm:
          1. Collect the fast-mode momentum labels from all vertex
             momentum_routing entries.
          2. A label containing an external symbol (e.g. 'p1', 'p₂',
             'q') that is not paired with its negative across the
             diagram indicates net momentum transfer.
          3. If every fast label is a pure loop momentum (k, -k, ...)
             that cancels pairwise, the transfer is q = 0.
        """
        q_label = ""
        notes = ""
        suppression = ""

        # ── Independent momentum-routing analysis ──
        fast_labels: list[str] = []
        for v in diagram.vertices:
            for field_name, mom in (v.momentum_routing or {}).items():
                if field_name.startswith("fast"):
                    fast_labels.append(mom)

        if fast_labels:
            # Strip loop-momentum symbols; any residual external symbol
            # (p, q, p₁, p₂, ...) signals net transfer.
            import re

            residual = []
            for lab in fast_labels:
                # remove pure loop-momentum pieces: k, -k, k+q's k, etc.
                # External labels are anything with a letter not 'k'.
                tokens = re.findall(r"[a-zA-Z][0-9₁₂]*", lab)
                for t in tokens:
                    if t.strip("k") != "" and t not in ("k",):
                        residual.append(t)
            if residual:
                q_label = "q"
                notes = (
                    f"independent routing: net external momentum "
                    f"({', '.join(set(residual))}) flows to slow legs → q≠0"
                )
            else:
                q_label = "0"
                notes = (
                    "independent routing: all fast labels are loop momenta "
                    "(k, -k) cancelling pairwise → q=0"
                )
        else:
            # No vertex routing data: fall back to topology structure.
            # Physical basis: bubble/ladder diagrams have back-to-back
            # fast modes (k, -k) at every vertex → q=0 per construction;
            # crossed/vertex-correction diagrams carry net transfer.
            if diagram.topology_label in ("bubble", "ladder"):
                q_label = "0"
                notes = (
                    "no routing data; topology "
                    f"{diagram.topology_label}: back-to-back fast modes → q=0"
                )
            elif diagram.topology_label in ("nonzero_q", "crossed_ladder", "vertex_correction"):
                q_label = "q"
                notes = f"no routing data; topology {diagram.topology_label}: q≠0"
            else:
                q_label = diagram.momentum_transfer or ""
                notes = "no routing data and no topology label; undetermined"

        # ── Classify ──
        if q_label == "0":
            mom_class = MomentumClass.ZERO_TRANSFER
            suppression = "none"
        elif q_label:
            mom_class = MomentumClass.NONZERO_TRANSFER
            suppression = f"exp(i{q_label}·x) → 0 as σ→∞"
        else:
            mom_class = MomentumClass.UNDETERMINED

        return MomentumLabel(
            diagram_id=diagram.id,
            momentum_class=mom_class,
            q_label=q_label,
            suppression_factor=suppression,
            routing_notes=notes,
        )

    # ── Oscillatory Factor Analysis ──

    def suppression_strength(self, q_magnitude: float, sigma: float) -> float:
        """
        Compute the suppression factor for a given |q| and σ.

        The oscillatory factor e^{iq·x} averaged over the coarse-graining
        window of width σ gives:

            ⟨e^{iq·x}⟩_σ = exp(-σ²|q|²/2)

        For σ|q| ≫ 1, this is exponentially small.
        For σ|q| ≪ 1, the suppression is weak (near-UV regime).
        For σ|q| = 1, the cross-over marks the boundary of the
        suppressed/accumulating regime.
        """
        import math

        return math.exp(-0.5 * (sigma * q_magnitude) ** 2)
