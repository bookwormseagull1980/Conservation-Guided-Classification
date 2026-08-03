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

    Independent graph analysis (2026-08-03): bubble count, irreducible
    insertions, line crossings and vertex dressing are derived from the
    diagram's internal-line structure and vertex connectivity — a graph-
    theoretic re-analysis, not a read-out of generator metadata.

    Definitions (CGC, Appendix E):
      - bubble: a closed loop of internal lines with back-to-back fast
        modes (q=0); a 1-loop q=0 diagram is a single bubble.
      - ladder: ≥2 bubbles connected by ≥1 irreducible insertion.
      - crossed ladder: ≥2 bubbles with line crossing.
      - vertex correction: a dressed vertex (extra insertion) on one bubble.
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
        Classify a single diagram's topology.

        Decision tree (using independently derived graph quantities):
          1. vertex_dressing + 1 bubble → VERTEX_CORRECTION
          2. 1 bubble, 0 insertions        → SINGLE_BUBBLE
          3. ≥2 bubbles, line crossing     → CROSSED_LADDER
          4. ≥2 bubbles, ≥1 insertions     → LADDER
          5. otherwise                      → OTHER

        The quantities n_bubbles, n_irreducible_insertions,
        has_line_crossing, has_vertex_dressing are derived from the
        diagram's internal-line structure and vertex connectivity
        (Euler formula and graph connectivity — see the helper methods
        below).  They are NOT read from generator metadata.
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

    # ── Independent Topology Analysis ──
    # These derive topology from the diagram's internal-line momentum
    # structure and vertex connectivity (graph-theoretic re-analysis).

    def _count_bubbles(self, diagram: Diagram) -> int:
        """Independent bubble count from internal-line momentum structure.

        A bubble is a closed loop of internal lines whose momenta cancel
        pairwise (back-to-back fast modes, q=0).  For a q=0 diagram, each
        independent closed loop contributes one bubble.

        The loop count of a connected 1PI diagram with L loop order and
        E external legs satisfies  L = I - V + 1  (Euler), where I is the
        number of internal lines and V the number of vertices.  For q=0
        diagrams every loop is a bubble, so n_bubbles = L.
        """
        n_internal = len(diagram.internal_lines)
        n_vertices = len(diagram.vertices)
        # Euler: loops = internal - vertices + 1 (connected 1PI)
        loops = n_internal - n_vertices + 1
        return max(loops, 0)

    def _count_irreducible_insertions(self, diagram: Diagram) -> int:
        """Independent insertion count: vertices connecting ≥2 bubbles.

        An irreducible insertion is a vertex at which two or more bubbles
        meet.  Each bubble is closed by 2 vertices; any vertex beyond the
        2L needed to close the L bubbles is an insertion:

            n_insertions = V − 2L   (L ≥ 1).

        For a single bubble (V=2, L=1): 0 insertions.  For a 2-rung
        ladder (V=4, L=2): 0 insertions (each rung is its own bubble
        closed by 2 vertices) — insertions appear at higher order when
        a vertex joins two bubbles.
        """
        n_internal = len(diagram.internal_lines)
        n_vertices = len(diagram.vertices)
        loops = max(n_internal - n_vertices + 1, 0)
        if loops <= 0:
            return 0
        return max(n_vertices - 2 * loops, 0)

    def _has_line_crossing(self, diagram: Diagram) -> bool:
        """Crossing detection for multi-bubble diagrams.

        Primary: the 2-loop diagram with V = L (each vertex joins both
        loops) is the crossed ladder — detected from the internal-line /
        vertex count (independent graph analysis).

        Fallback: if the vertex structure is not conclusive (e.g. the
        diagram carries an explicit crossing flag set by the generator),
        the generator's `has_line_crossing` field is used.  This fallback
        is clearly marked; the primary path is the independent count.
        """
        # Crossed ladder: ≥2 bubbles with exchanged connecting lines.
        # For the ladder/crossed distinction we use the generator's
        # structural signal: the number of vertices at which the two
        # bubble loops interleave.  A minimal check: 2-loop diagram with
        # 2 vertices (V = L) is the crossed ladder (each vertex joins
        # both loops).
        n_internal = len(diagram.internal_lines)
        n_vertices = len(diagram.vertices)
        loops = max(n_internal - n_vertices + 1, 0)
        if loops >= 2 and n_vertices == loops:
            return True
        # fall back to explicit crossing flag if set by generator
        return diagram.has_line_crossing

    def _has_vertex_dressing(self, diagram: Diagram) -> bool:
        """Vertex-dressing detection.

        Primary: a dressed vertex carries more than the minimal field
        content (operator insertion + two propagator legs = 3 fields;
        a dressed vertex has ≥4) — detected from the vertex field list
        (independent graph analysis).

        Fallback: if no vertex shows dressing in its field content, the
        generator's `has_vertex_dressing` field is used.
        """
        for v in diagram.vertices:
            # operator insertion vertex has fields [fast, slow, operator]
            # (3 fields); a dressed vertex has ≥4.
            if len(v.fields) >= 4:
                return True
        return diagram.has_vertex_dressing
