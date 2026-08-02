"""
Topology Classifier — Phase 2.3
============================================================================

Within the q=0 class (diagrams that survive momentum transfer filtering),
further classify by topological structure:

  - SINGLE_BUBBLE:  one connected loop with operator insertion, no
    irreducible internal vertices → contributes to CONTINUUM (ρ_cont)
  - LADDER:         multiple bubbles connected by irreducible vertices V
    → ACCUMULATING class, only topology capable of producing a δ-pole
  - CROSSED_LADDER: bubbles with crossed internal lines (non-planar)
  - VERTEX_CORRECTION: vertex dressings on the operator insertion
  - OTHER:          any topology not fitting the above

The topological classification is essential because it distinguishes
between diagrams that:
  (a) only renormalize the continuum (single bubble)
  (b) can build up a geometric series → pole (ladder)
  (c) may contribute but not in a ladder-resummable way (vertex correction, etc.)

This is the CGC method's second classification axis, complementary to
momentum transfer. Together they form the "classification" step of
"classify, filter, resummate."

Verification target (Phase 2.6 benchmark):
  For Tμν spin-2 channel, topology classification must match
  Appendix E Figures 1 and 3.
"""


# References
#     Diagram topology (bubble vs ladder): standard perturbation theory
#     Resummation: ladder approximation (Roberts-Williams 1994)
#

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from .diagram_generator import Diagram
from .momentum_classifier import MomentumClassification

# ── Topology Types ───────────────────────────────────────────────────────


class TopologyClass(Enum):
    """Topological structure of a q=0 diagram."""

    SINGLE_BUBBLE = auto()  # one bubble, no irreducible insertion
    LADDER = auto()  # multiple bubbles + irreducible V insertions
    CROSSED_LADDER = auto()  # non-planar crossing of internal lines
    VERTEX_CORRECTION = auto()  # vertex dressing at operator insertion
    OTHER = auto()  # any other topology


@dataclass
class TopologyLabel:
    """Topology classification for one diagram."""

    diagram_id: str
    topology_class: TopologyClass
    n_bubbles: int = 1  # number of closed loops
    n_insertions: int = 0  # number of irreducible V vertices
    is_ladder_resummable: bool = False
    contributes_to: str = "continuum"  # "continuum" | "pole" | "unknown"
    notes: str = ""


@dataclass
class TopologyClassification:
    """Complete topology classification for q=0 diagrams."""

    operator_name: str
    single_bubble: list[TopologyLabel] = field(default_factory=list)
    ladder: list[TopologyLabel] = field(default_factory=list)
    crossed_ladder: list[TopologyLabel] = field(default_factory=list)
    vertex_correction: list[TopologyLabel] = field(default_factory=list)
    other: list[TopologyLabel] = field(default_factory=list)

    @property
    def accumulating(self) -> list[TopologyLabel]:
        """Diagrams that can accumulate → potential pole formation."""
        return self.ladder

    @property
    def continuum_only(self) -> list[TopologyLabel]:
        """Diagrams contributing only to the continuum part."""
        return self.single_bubble

    @property
    def needs_analysis(self) -> list[TopologyLabel]:
        """Diagrams requiring further analysis (non-standard topologies)."""
        return self.crossed_ladder + self.vertex_correction + self.other

    def summary(self) -> str:
        lines = [
            f"Topology Classification (q=0 only): {self.operator_name}",
            f"  single bubble     (continuum): {len(self.single_bubble)}",
            f"  ladder            (polarizable): {len(self.ladder)}",
            f"  crossed ladder    (to analyze): {len(self.crossed_ladder)}",
            f"  vertex correction (to analyze): {len(self.vertex_correction)}",
            f"  other             (to analyze): {len(self.other)}",
        ]
        return "\n".join(lines)


# ── Classifier ───────────────────────────────────────────────────────────


class TopologyClassifier:
    """
    Classifies q=0 diagrams by topological structure.

    Current implementation: reads pre-computed topology metadata
    (n_bubbles, n_irreducible_insertions, etc.) from Diagram fields
    set by the diagram generator. Independent graph-theoretic analysis
    (counting closed loops from adjacency, detecting irreducible
    vertices by cut-set analysis) is deferred.

    This means the classifier's correctness depends on the generator
    producing accurate topology metadata. For the Tμν spin-2 benchmark,
    the metadata is verified against Appendix E.
    """

    def classify(self, mom_class: MomentumClassification, diagrams: list[Diagram]) -> TopologyClassification:
        """
        Classify all q=0 diagrams by topology.

        Args:
            mom_class: momentum classification result
            diagrams: original diagram list (for topology metadata)

        Returns:
            TopologyClassification
        """
        topo = TopologyClassification(operator_name=mom_class.operator_name)

        # Build lookup: diagram_id → Diagram
        diagram_map: dict[str, Diagram] = {d.id: d for d in diagrams}

        for label in mom_class.zero_transfer:
            diag = diagram_map.get(label.diagram_id)
            if diag is None:
                continue

            topo_label = self._classify_one(diag)

            if topo_label.topology_class == TopologyClass.SINGLE_BUBBLE:
                topo.single_bubble.append(topo_label)
            elif topo_label.topology_class == TopologyClass.LADDER:
                topo.ladder.append(topo_label)
            elif topo_label.topology_class == TopologyClass.CROSSED_LADDER:
                topo.crossed_ladder.append(topo_label)
            elif topo_label.topology_class == TopologyClass.VERTEX_CORRECTION:
                topo.vertex_correction.append(topo_label)
            else:
                topo.other.append(topo_label)

        return topo

    def _classify_one(self, diagram: Diagram) -> TopologyLabel:
        """
        Classify a single diagram's topology using pre-computed metadata.

        Decision tree:
          1. vertex_dressing + 1 bubble → VERTEX_CORRECTION
          2. 1 bubble, 0 insertions        → SINGLE_BUBBLE
          3. ≥2 bubbles, line crossing     → CROSSED_LADDER
          4. ≥2 bubbles, ≥1 insertions     → LADDER
          5. otherwise                      → OTHER

        The metadata values (n_bubbles, n_irreducible_insertions,
        has_line_crossing, has_vertex_dressing) are expected to be
        set by the diagram generator. This classifier does not
        independently derive them from the graph structure.
        """
        n_bubbles = self._count_bubbles(diagram)
        n_insertions = self._count_irreducible_insertions(diagram)
        has_crossing = self._has_line_crossing(diagram)
        has_vertex_dressing = self._has_vertex_dressing(diagram)

        # ── Topology decision ──
        if has_vertex_dressing and n_bubbles == 1:
            topo_class = TopologyClass.VERTEX_CORRECTION
            contributes_to = "unknown"
            is_ladder = False
        elif n_bubbles == 1 and n_insertions == 0:
            topo_class = TopologyClass.SINGLE_BUBBLE
            contributes_to = "continuum"
            is_ladder = False
        elif n_bubbles >= 2 and has_crossing:
            topo_class = TopologyClass.CROSSED_LADDER
            contributes_to = "unknown"
            is_ladder = False
        elif n_bubbles >= 2 and n_insertions >= 1:
            topo_class = TopologyClass.LADDER
            contributes_to = "pole"
            is_ladder = True
        else:
            topo_class = TopologyClass.OTHER
            contributes_to = "unknown"
            is_ladder = False

        return TopologyLabel(
            diagram_id=diagram.id,
            topology_class=topo_class,
            n_bubbles=n_bubbles,
            n_insertions=n_insertions,
            is_ladder_resummable=is_ladder,
            contributes_to=contributes_to,
        )

    # ── Topology Metadata Helpers ──
    # These read pre-computed fields from the Diagram object.
    # Independent graph analysis (loop counting from adjacency,
    # cut-set irreducibility detection) is NOT implemented here.

    def _count_bubbles(self, diagram: Diagram) -> int:
        """Return n_bubbles from generator metadata.

        A 'bubble' in CGC is a closed loop with back-to-back fast modes
        (q=0). Nonzero-q loops are NOT bubbles — they have n_bubbles=0
        even if they contain a loop.

        If n_bubbles is 0 (not set or explicitly zero), returns 0.
        Does NOT fall back to loop_number — that would misclassify
        ladder diagrams as single-bubble.
        """
        return diagram.n_bubbles

    def _count_irreducible_insertions(self, diagram: Diagram) -> int:
        """Return n_irreducible_insertions from generator metadata.

        An 'irreducible insertion' is a vertex V that connects two or more
        independent bubbles. Single-bubble and nonzero-q diagrams have 0.
        """
        return diagram.n_irreducible_insertions

    def _has_line_crossing(self, diagram: Diagram) -> bool:
        """Return has_line_crossing from generator metadata."""
        return diagram.has_line_crossing

    def _has_vertex_dressing(self, diagram: Diagram) -> bool:
        """Return has_vertex_dressing from generator metadata."""
        return diagram.has_vertex_dressing
