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

        Priority order:
          1. Explicit momentum_transfer field set by diagram generator
             (the generator is assumed to have correctly analyzed momentum
             routing — this is an architecture choice, not a dynamic
             re-derivation)
          2. Topology-label-based fallback (only when momentum_transfer
             is None):
             - bubble → q=0 (fast modes back-to-back by construction)
             - ladder → q=0 (each bubble has back-to-back fast modes)
             - nonzero_q, crossed_ladder, vertex_correction → q≠0

        Limitation: the current implementation reads pre-computed metadata
        rather than independently analyzing momentum routing from vertex
        and propagator data. A full momentum-routing analyzer (constructing
        q from the fast→slow momentum flows at each vertex) is deferred.
        """
        # ── Determine momentum transfer ──
        q_label = ""
        suppression = ""
        notes = ""

        # Priority 1: explicit momentum_transfer set by generator
        if diagram.momentum_transfer:
            q_label = diagram.momentum_transfer
            notes = "explicit q=0 from generator" if q_label == "0" else f"explicit q={q_label} from generator"

        # Priority 2 (fallback): infer from topology label
        # Only used when momentum_transfer is not explicitly set
        if not q_label:
            if diagram.topology_label in ("bubble",):
                q_label = "0"
                notes = "single bubble: external legs back-to-back → q=0"
            elif diagram.topology_label in ("ladder",):
                # Ladder with q=0: fast k and -k back-to-back in each bubble
                q_label = "0"
                notes = "ladder: fast modes back-to-back → q=0 per bubble"
            elif diagram.topology_label in ("nonzero_q", "crossed_ladder", "vertex_correction"):
                q_label = "q"
                notes = f"topology {diagram.topology_label}: nonzero momentum transfer"

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
