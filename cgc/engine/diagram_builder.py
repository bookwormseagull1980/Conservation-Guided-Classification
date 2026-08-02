"""Self-contained diagram builder — adjacency-list + vertex-based enumeration.

Key design decisions (per user specification):
  1. Adjacency-list representation: vertices + edges + external legs
  2. Vertex-based enumeration from SM.mod's 85 physical vertices
  3. Graph isomorphism checking for deduplication
  4. Specific two-loop topologies (crossed ladder, vertex correction)
  5. JSON exchange format for inter-module communication
  6. Zero external dependencies (no QGRAF, no Fortran, pure Python)

Author: CGC Phase 2, self-contained rewrite
Date: 2026-07-29
"""


# References
#     Feynman diagram topology: one-loop bubble/triangle/box classification
#     Momentum routing: standard QFT textbook (Peskin & Schroeder, Ch. 7)
#

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .diagram_generator import Diagram

# ═══════════════════════════════════════════════════════════════
# 1. Adjacency-list diagram representation
# ═══════════════════════════════════════════════════════════════


class FieldKind(str, Enum):
    """Propagator / field type."""

    FERMION = "F"
    SCALAR = "S"
    VECTOR = "V"
    ANTI_FERMION = "-F"
    ANTI_SCALAR = "-S"
    ANTI_VECTOR = "-V"


@dataclass
class AdjacencyVertex:
    """A vertex in the adjacency-list diagram.

    Each vertex corresponds to either:
    - A CGC operator insertion (type="CGC_op") — external momentum in/out
    - An SM interaction from SM.mod (type="SM") — 3-pt or 4-pt coupling
    """

    vid: int  # unique vertex ID
    vtype: str  # "CGC_op" or "SM"
    sm_vertex_index: int = -1  # index into SM.mod's 136 vertices (if vtype="SM")
    fields_in: list[str] = field(default_factory=list)  # incoming field types
    fields_out: list[str] = field(default_factory=list)  # outgoing field types
    coupling: str = ""  # coupling expression from SM.mod


@dataclass
class AdjacencyEdge:
    """An edge (propagator) in the adjacency-list diagram."""

    eid: int  # unique edge ID
    vid_from: int  # source vertex ID
    vid_to: int  # target vertex ID
    field_type: str  # "F", "S", "V" (+ possible indices like "F[1]")
    is_external: bool = False  # is this an external leg?
    is_fast_mode: bool = False  # carries the fast-mode momentum?
    momentum_label: str = ""  # "q=0" or "q_nonzero" or external momentum label


@dataclass
class AdjacencyDiagram:
    """Full diagram in adjacency-list format.

    Self-contained representation that can be:
    - Serialized to/from JSON
    - Checked for isomorphism
    - Converted to/from the existing CGC Diagram dataclass
    """

    diagram_id: str = ""  # content hash for unique identification
    operator_name: str = ""  # which CGC operator
    loop_order: int = 1  # 1 = one-loop, 2 = two-loop
    vertices: list[AdjacencyVertex] = field(default_factory=list)
    edges: list[AdjacencyEdge] = field(default_factory=list)
    external_legs: list[int] = field(default_factory=list)  # edge IDs of external legs

    def compute_id(self) -> str:
        """Content-based hash for deduplication."""
        # Build canonical representation
        vertices_repr = tuple(
            (v.vid, v.vtype, tuple(sorted(v.fields_in)), tuple(sorted(v.fields_out)))
            for v in sorted(self.vertices, key=lambda x: x.vid)
        )
        edges_repr = tuple(
            (e.eid, e.vid_from, e.vid_to, e.field_type, e.is_external) for e in sorted(self.edges, key=lambda x: x.eid)
        )
        canonical = json.dumps({"v": vertices_repr, "e": edges_repr}, sort_keys=True)
        self.diagram_id = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        return self.diagram_id

    def degree_sequence(self) -> tuple[int, ...]:
        """Sorted degree sequence of internal vertices (for isomorphism pre-filter)."""
        degrees: dict[int, int] = {}
        for e in self.edges:
            if e.is_external:
                continue
            degrees[e.vid_from] = degrees.get(e.vid_from, 0) + 1
            degrees[e.vid_to] = degrees.get(e.vid_to, 0) + 1
        return tuple(sorted(degrees.values()))

    def edge_type_sequence(self) -> tuple[str, ...]:
        """Sorted edge field types (for isomorphism pre-filter)."""
        return tuple(sorted(e.field_type for e in self.edges if not e.is_external))

    def to_json(self) -> str:
        """Serialize to JSON (per spec)."""
        return json.dumps(
            {
                "diagram_id": self.compute_id(),
                "operator_name": self.operator_name,
                "loop_order": self.loop_order,
                "vertices": [
                    {
                        "vid": v.vid,
                        "vtype": v.vtype,
                        "sm_vertex_index": v.sm_vertex_index,
                        "fields_in": v.fields_in,
                        "fields_out": v.fields_out,
                        "coupling": v.coupling,
                    }
                    for v in self.vertices
                ],
                "edges": [
                    {
                        "eid": e.eid,
                        "vid_from": e.vid_from,
                        "vid_to": e.vid_to,
                        "field_type": e.field_type,
                        "is_external": e.is_external,
                        "is_fast_mode": e.is_fast_mode,
                        "momentum_label": e.momentum_label,
                    }
                    for e in self.edges
                ],
                "external_legs": self.external_legs,
            },
            indent=2,
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, json_str: str) -> AdjacencyDiagram:
        """Deserialize from JSON."""
        d = json.loads(json_str)
        diag = cls(
            diagram_id=d.get("diagram_id", ""),
            operator_name=d.get("operator_name", ""),
            loop_order=d.get("loop_order", 1),
            external_legs=d.get("external_legs", []),
        )
        diag.vertices = [
            AdjacencyVertex(
                vid=v["vid"],
                vtype=v["vtype"],
                sm_vertex_index=v.get("sm_vertex_index", -1),
                fields_in=v.get("fields_in", []),
                fields_out=v.get("fields_out", []),
                coupling=v.get("coupling", ""),
            )
            for v in d.get("vertices", [])
        ]
        diag.edges = [
            AdjacencyEdge(
                eid=e["eid"],
                vid_from=e["vid_from"],
                vid_to=e["vid_to"],
                field_type=e.get("field_type", ""),
                is_external=e.get("is_external", False),
                is_fast_mode=e.get("is_fast_mode", False),
                momentum_label=e.get("momentum_label", ""),
            )
            for e in d.get("edges", [])
        ]
        return diag


# ═══════════════════════════════════════════════════════════════
# 2. Graph isomorphism checker
# ═══════════════════════════════════════════════════════════════


def are_isomorphic(d1: AdjacencyDiagram, d2: AdjacencyDiagram) -> bool:
    """Check whether two adjacency diagrams are topologically equivalent.

    Strategy (layered):
      1. Quick pre-filters: degree sequence, edge type sequence, vertex count
      2. If all match, do full adjacency matrix comparison with permutations
      3. This is O(V!) in worst case but V ≤ ~10 for CGC diagrams
    """
    # Pre-filter 1: same number of internal vertices and edges
    v1_int = [v for v in d1.vertices if v.vtype != "CGC_op"]
    v2_int = [v for v in d2.vertices if v.vtype != "CGC_op"]
    if len(v1_int) != len(v2_int):
        return False

    e1_int = [e for e in d1.edges if not e.is_external]
    e2_int = [e for e in d2.edges if not e.is_external]
    if len(e1_int) != len(e2_int):
        return False

    # Pre-filter 2: degree sequences
    if d1.degree_sequence() != d2.degree_sequence():
        return False

    # Pre-filter 3: edge type sequences
    if d1.edge_type_sequence() != d2.edge_type_sequence():
        return False

    # Pre-filter 4: vertex type signatures (SM vertex indices sorted)
    sig1 = tuple(sorted(v.sm_vertex_index for v in v1_int))
    sig2 = tuple(sorted(v.sm_vertex_index for v in v2_int))
    if sig1 != sig2:
        return False

    # Full check: build adjacency matrices and try all permutations
    return _adjacency_matrix_isomorphic(d1, d2)


def _adjacency_matrix_isomorphic(d1: AdjacencyDiagram, d2: AdjacencyDiagram) -> bool:
    """Build adjacency matrices and check for permutation equivalence."""
    import itertools

    v1_int = [v for v in d1.vertices if v.vtype != "CGC_op"]
    v2_int = [v for v in d2.vertices if v.vtype != "CGC_op"]
    n = len(v1_int)

    if n <= 1:
        return True  # 0 or 1 internal vertices: trivial

    # Build labeled adjacency matrix for d1
    def build_adj(diag: Any, v_int: list[Any]) -> dict[Any, Any]:
        vid_to_idx = {v.vid: i for i, v in enumerate(v_int)}
        adj = {}
        for e in diag.edges:
            if e.is_external:
                continue
            if e.vid_from in vid_to_idx and e.vid_to in vid_to_idx:
                i = vid_to_idx[e.vid_from]
                j = vid_to_idx[e.vid_to]
                adj[(i, j)] = e.field_type
                adj[(j, i)] = e.field_type  # undirected for topology
        return adj

    adj1 = build_adj(d1, v1_int)
    adj2 = build_adj(d2, v2_int)

    # Vertex signatures: (degree, vertex_type_index, neighbor types)
    def vertex_sig(diag: Any, v_int: list[Any], adj: dict[Any, Any]) -> list[Any]:
        sigs = []
        for v in v_int:
            idx = next(i for i, vi in enumerate(v_int) if vi.vid == v.vid)
            degree = sum(1 for (i, j) in adj if i == idx)
            neighbors = tuple(sorted(adj[(idx, j)] for (i, j) in adj if i == idx))
            sigs.append((degree, v.sm_vertex_index, neighbors))
        return sigs

    sig1 = vertex_sig(d1, v1_int, adj1)
    sig2 = vertex_sig(d2, v2_int, adj2)

    if sorted(sig1) != sorted(sig2):
        return False

    # For n <= 6, try all permutations
    if n > 6:
        # Fall back to signature-based comparison for larger graphs
        return sorted(sig1) == sorted(sig2)

    perm_indices = list(range(n))
    for perm in itertools.permutations(perm_indices):
        match = True
        for i in range(n):
            for j in range(i + 1, n):
                t1 = adj1.get((i, j))
                t2 = adj2.get((perm[i], perm[j]))
                if t1 != t2:
                    match = False
                    break
            if not match:
                break
        if match:
            return True

    return sorted(sig1) == sorted(sig2)


def deduplicate_diagrams(diagrams: list[AdjacencyDiagram]) -> list[AdjacencyDiagram]:
    """Remove topologically equivalent diagrams.

    Uses isomorphism check. First diagram of each equivalence class is kept.
    Diagrams with different external momentum labels (q=0 vs q≠0) are NOT
    considered equivalent even if topologically identical.
    """

    def _momentum_signature(d: AdjacencyDiagram) -> tuple[str, ...]:
        """Unique signature from external momentum labels."""
        return tuple(sorted(e.momentum_label for e in d.edges if e.is_external))

    unique: list[Any] = []
    for d in diagrams:
        q_sig = _momentum_signature(d)
        is_dup = False
        for u in unique:
            if _momentum_signature(u) != q_sig:
                continue
            if are_isomorphic(d, u):
                is_dup = True
                break
        if not is_dup:
            unique.append(d)
    return unique


# ═══════════════════════════════════════════════════════════════
# 3. Vertex-based diagram enumerator
# ═══════════════════════════════════════════════════════════════


@dataclass
class SMVertexInfo:
    """Processed SM vertex information for enumeration."""

    index: int  # original SM.mod index
    fields: list[str]  # all fields (in+out, using "-" for anti)
    coupling: str  # coupling expression
    n_legs: int  # total legs
    field_types: list[str]  # simplified: "F", "S", "V" (with anti marker)


class DiagramEnumerator:
    """Enumerate one-loop and two-loop diagrams from SM.mod vertices.

    Implements the vertex-based enumeration strategy:
      1. For a given CGC operator, find matching SM vertices
      2. Pair external legs to vertices
      3. Close remaining legs into internal propagators
      4. Deduplicate via graph isomorphism
    """

    def __init__(self) -> None:
        """Initialize from SM.mod vertex data."""
        from .fa_model_parser import load_sm_model

        self.sm_model = load_sm_model()
        self._build_vertex_index()

    def _build_vertex_index(self) -> None:
        """Index SM vertices by field type for fast lookup."""
        self.physical_vertices: list[SMVertexInfo] = []
        self.by_field_type: dict[str, list[int]] = {}

        for idx, v in enumerate(self.sm_model.vertices):
            n = len(v.fields)
            # Skip counterterms (1-point or dZxxx couplings)
            if n <= 1:
                continue
            if str(v.coupling_name).startswith("d"):
                continue
            if any(f.strip().startswith("U[") for f in v.fields):
                continue  # ghost vertices

            field_types = []
            for f in v.fields:
                clean = f.strip()
                ft = clean[0:2] if clean.startswith("-") else clean[0]  # "-F"/"F", etc.
                field_types.append(ft)

            info = SMVertexInfo(
                index=idx,
                fields=[f.strip() for f in v.fields],
                coupling=str(v.coupling_name),
                n_legs=n,
                field_types=field_types,
            )
            self.physical_vertices.append(info)

            # Index by field type patterns
            for ft in field_types:
                if ft not in self.by_field_type:
                    self.by_field_type[ft] = []
                self.by_field_type[ft].append(len(self.physical_vertices) - 1)

    def _find_representative_vertex(
        self, field_type: str, min_legs: int = 1, total_legs: int = 0
    ) -> SMVertexInfo | None:
        """Find a representative SM vertex matching field type requirements.

        For bubble diagrams: need at least min_legs (for external coupling) PLUS
        additional legs for the internal loop. Pass total_legs=4 for bubble.
        For two-vertex diagrams: each vertex needs min_legs for external coupling
        and additional legs to connect, pass total_legs=3.

        Prefers vertices where both field and anti-field appear (allows self-loop).
        """
        base = field_type.lstrip("-")
        anti = "-" + base

        # First: vertex with field+anti AND enough total legs
        best = None
        for vinfo in self.physical_vertices:
            has_base = sum(1 for ft in vinfo.field_types if ft == base)
            has_anti = sum(1 for ft in vinfo.field_types if ft == anti)
            total_match = has_base + has_anti
            if total_match >= min_legs and vinfo.n_legs >= total_legs:
                if has_base >= 1 and has_anti >= 1:
                    return vinfo  # perfect: field+anti, enough legs
                if best is None:
                    best = vinfo

        if best is not None:
            return best

        # Fallback: any vertex with enough legs of this type
        for vinfo in self.physical_vertices:
            count = sum(1 for ft in vinfo.field_types if ft in (base, anti))
            if count >= min_legs and vinfo.n_legs >= total_legs:
                return vinfo

        # Last resort: enough matching legs, ignore total_legs
        for vinfo in self.physical_vertices:
            count = sum(1 for ft in vinfo.field_types if ft in (base, anti))
            if count >= min_legs:
                return vinfo

        return None

    def _get_field_type(self, field_str: str) -> str:
        """Map SM field string to simplified type: F, V, S."""
        clean = field_str.strip().lstrip("-")
        if clean.startswith("F"):
            return "F"
        if clean.startswith("V"):
            return "V"
        if clean.startswith("S"):
            return "S"
        return "?"

    def _loop_field_types(self, operator: Any) -> list[str]:
        """Return the list of field TYPES that can run in the loop for this operator.

        These are categories F, V, S (not specific particle species).
        One diagram is generated per (field_type, kinematic_class) pair.
        """
        name = operator.op_type.name if hasattr(operator, "op_type") else ""

        if "CONSERVED_CURRENT" in name:
            # Tμν: universal — all SM fields couple
            return ["F", "V", "S"]

        if "GAUGE_FIELD_STRENGTH" in name:
            # F²: gauge bosons + charged fermions
            return ["V", "F"]

        if "UNPROTECTED_FERMION" in name:
            # ψ̄ψ: fermions only
            return ["F"]

        if "UNPROTECTED_SCALAR" in name:
            # λφ⁴: scalars only (4-external-leg operator)
            return ["S"]

        return []

    def enumerate_one_loop(self, operator: Any) -> list[AdjacencyDiagram]:
        """Enumerate one-loop diagrams at field-TYPE level.

        One diagram per (loop_field_type, kinematic_class) pair.
        Uses representative SM vertices for metadata but builds ABSTRACT
        topologies (2 SM vertices + internal propagator loop) — this
        captures the CGC classification without enumerating all SM species.
        """
        diagrams: list[AdjacencyDiagram] = []
        operator_name = operator.name if hasattr(operator, "name") else str(operator)

        # Handle 4-point operator (λφ⁴) specially
        if hasattr(operator, "op_type"):
            from cgc.engine.diagram_generator import OperatorType

            if operator.op_type == OperatorType.UNPROTECTED_SCALAR and operator.external_momenta == 4:
                return self._enumerate_scalar_quartic(operator)

        # Standard 2-external-leg operators
        loop_types = self._loop_field_types(operator)
        if not loop_types:
            return diagrams

        for ft in loop_types:
            base = ft.lstrip("-")

            # Find a representative SM vertex (just for bookkeeping)
            rep_v = self._find_representative_vertex(base, min_legs=1, total_legs=0)
            rep_index = rep_v.index if rep_v else -1
            rep_coupling = rep_v.coupling if rep_v else ""

            # Build TWO diagrams per field type: q=0 (bubble) and q≠0
            for q0 in [True, False]:
                d = self._build_abstract_loop_diagram(operator_name, ft, rep_index, rep_coupling, q0)
                diagrams.append(d)

        return deduplicate_diagrams(diagrams)

    def _build_abstract_loop_diagram(
        self, operator_name: str, loop_ft: str, sm_index: int, sm_coupling: str, q0: bool
    ) -> AdjacencyDiagram:
        """Build an abstract CGC diagram for a field-type loop.

        Topology: CGC_op → 2 SM vertices with internal propagator loop.
        The SM vertex metadata is for bookkeeping only; the topology
        is what matters for momentum/topology classification.
        """
        base = loop_ft.lstrip("-")

        vertices = [
            AdjacencyVertex(vid=0, vtype="CGC_op"),
            AdjacencyVertex(vid=1, vtype="SM", sm_vertex_index=sm_index, coupling=sm_coupling),
            AdjacencyVertex(vid=2, vtype="SM", sm_vertex_index=sm_index, coupling=sm_coupling),
        ]

        edges = [
            AdjacencyEdge(
                eid=0,
                vid_from=0,
                vid_to=1,
                field_type=base,
                is_external=True,
                momentum_label="q=0" if q0 else "q_nonzero",
            ),
            AdjacencyEdge(
                eid=1,
                vid_from=0,
                vid_to=2,
                field_type=base,
                is_external=True,
                momentum_label="q=0" if q0 else "q_nonzero",
            ),
            AdjacencyEdge(eid=2, vid_from=1, vid_to=2, field_type=base, is_external=False, is_fast_mode=True),
        ]

        d = AdjacencyDiagram(
            operator_name=f"{operator_name}/{loop_ft}-loop",
            loop_order=1,
            vertices=vertices,
            edges=edges,
            external_legs=[0, 1],
        )
        d.compute_id()
        return d

    def _enumerate_scalar_quartic(self, operator: Any) -> list[AdjacencyDiagram]:
        """Generate 4 diagrams for scalar quartic operator (λφ⁴, 4-ext legs).

        Topology: 4 external scalar legs connect to SM vertices.
        Expected: 4 diagrams total (2 bubble + 2 non-bubble).
        """
        diagrams: list[AdjacencyDiagram] = []
        operator_name = operator.name if hasattr(operator, "name") else str(operator)

        # Use representative scalar vertex (e.g., SSSS or SSV or FFS)
        vinfo = self._find_representative_vertex("S", min_legs=4, total_legs=4)
        if not vinfo:
            # Fallback: any 4-point vertex
            for v in self.physical_vertices:
                if v.n_legs >= 4:
                    vinfo = v
                    break
        if not vinfo:
            return diagrams

        # 4 external legs

        # Diagram 1-2: bubble = ext legs (e1,e2) on one vertex, (e3,e4) on another
        # Both vertices are the same SM vertex type, connected by internal propagators
        for pairing_id in range(2):
            if pairing_id == 0:
                pair_a = (0, 1)  # e1+e2 on vertex A
                pair_b = (2, 3)  # e3+e4 on vertex B
            else:
                pair_a = (0, 2)  # e1+e3 on vertex A
                pair_b = (1, 3)  # e2+e4 on vertex B

            diag = self._build_scalar_quartic_diagram(vinfo, pair_a, pair_b, operator_name, q0=(pairing_id == 0))
            diagrams.append(diag)

        # Diagram 3-4: non-bubble topologies
        # Crossed external leg assignment
        for pairing_id in range(2, 4):
            if pairing_id == 2:
                pair_a = (0, 3)
                pair_b = (1, 2)
            else:
                pair_a = (0, 1)
                pair_b = (2, 3)  # different internal routing

            diag = self._build_scalar_quartic_diagram(vinfo, pair_a, pair_b, operator_name, q0=False)
            diagrams.append(diag)

        return diagrams

    def _build_scalar_quartic_diagram(
        self,
        vinfo: SMVertexInfo,
        ext_pair_a: tuple[int, int],
        ext_pair_b: tuple[int, int],
        operator_name: str,
        q0: bool = True,
    ) -> AdjacencyDiagram:
        """Build a 4-point scalar quartic adjacency diagram."""
        # 3 vertices: 0=CGC_op, 1=SM vertex A, 2=SM vertex B
        vertices = [
            AdjacencyVertex(vid=0, vtype="CGC_op"),
            AdjacencyVertex(
                vid=1, vtype="SM", sm_vertex_index=vinfo.index, fields_in=vinfo.fields, coupling=vinfo.coupling
            ),
            AdjacencyVertex(
                vid=2, vtype="SM", sm_vertex_index=vinfo.index, fields_in=vinfo.fields, coupling=vinfo.coupling
            ),
        ]

        edges: list[Any] = []
        # 4 external edges: each from CGC_op to an SM vertex
        for _idx, (pair, vid) in enumerate([(ext_pair_a, 1), (ext_pair_b, 2)]):
            for _leg in pair:
                edges.append(
                    AdjacencyEdge(
                        eid=len(edges),
                        vid_from=0,
                        vid_to=vid,
                        field_type="S",
                        is_external=True,
                        momentum_label=f"q={0 if q0 else 'nonzero'}",
                    )
                )

        # Internal propagator between SM vertices (if applicable)
        # For bubble (q=0): one internal propagator
        # For non-bubble: topology differs
        internal_eid = len(edges)
        edges.append(
            AdjacencyEdge(
                eid=internal_eid,
                vid_from=1,
                vid_to=2,
                field_type="S",
                is_external=False,
                is_fast_mode=True,
            )
        )

        d = AdjacencyDiagram(
            operator_name=operator_name,
            loop_order=1,
            vertices=vertices,
            edges=edges,
            external_legs=[0, 1, 2, 3],
        )
        d.compute_id()
        return d

    def _external_field_types(self, operator: Any) -> list[str]:
        """Determine external leg field types from operator specification."""
        name = operator.op_type.name if hasattr(operator, "op_type") else ""

        # All SM fields couple to Tμν (universality of gravity)
        if "CONSERVED_CURRENT" in name:
            return ["F", "F"]  # placeholder — all fields couple

        # Gauge field strength: couples to gauge bosons and fermions
        if "GAUGE_FIELD_STRENGTH" in name:
            return ["V", "V"]  # external gauge legs

        # Fermion bilinear
        if "FERMION" in name:
            return ["F", "F"]

        # Scalar quartic
        if "SCALAR" in name:
            return ["S", "S"]

        return ["?", "?"]

    def _count_matching_legs(self, vinfo: SMVertexInfo, type1: str, type2: str) -> int:
        """Count how many vertex legs match the given external types."""
        count = 0
        types_needed = [t for t in (type1, type2) if t]
        remaining = list(types_needed)
        used = set()
        for i, ft in enumerate(vinfo.field_types):
            for t in remaining:
                if i not in used and self._type_matches(ft, t):
                    used.add(i)
                    remaining.remove(t)
                    count += 1
                    break
        return count

    def _type_matches(self, vertex_type: str, ext_type: str) -> bool:
        """Check if a vertex field type is compatible with an external leg type."""
        if not ext_type:
            return True
        # Strip anti-particle marker
        vt = vertex_type.lstrip("-")
        et = ext_type.lstrip("-")
        return vt.startswith(et) or et.startswith(vt)

    def _close_into_internal_loop(        self, vinfo: SMVertexInfo, ext_i: int, ext_j: int, remaining: list[tuple[int, str]], operator: Any
    ) -> list[AdjacencyDiagram]:
        """Close remaining legs of a single vertex into an internal loop.

        For Type A bubble: remaining legs must pair into compatible propagators.
        """
        diagrams: list[Any] = []
        n_rem = len(remaining)

        if n_rem == 0:
            # 2-point vertex: both legs are external, no internal loop
            # This is a tree-level diagram, not one-loop
            return diagrams

        if n_rem == 2:
            # Single internal propagator: connect rem[0] ↔ rem[1]
            (k1, ft1), (k2, ft2) = remaining[0], remaining[1]
            if self._can_form_propagator(ft1, ft2):
                d = self._build_bubble_diagram(vinfo, ext_i, ext_j, [(k1, k2, ft1)], operator, q0=True)
                diagrams.append(d)

        elif n_rem == 4:
            # Two internal propagators (vertex is 4-pt with 2 external)
            # Try all pairings
            idxs = [k for k, _ in remaining]
            fts = [ft for _, ft in remaining]
            for a in range(4):
                for b in range(a + 1, 4):
                    pair1 = [a, b]
                    pair2 = [x for x in range(4) if x not in pair1]
                    if self._can_form_propagator(fts[pair1[0]], fts[pair1[1]]) and self._can_form_propagator(
                        fts[pair2[0]], fts[pair2[1]]
                    ):
                        edges = [
                            (idxs[pair1[0]], idxs[pair1[1]], fts[pair1[0]]),
                            (idxs[pair2[0]], idxs[pair2[1]], fts[pair2[0]]),
                        ]
                        d = self._build_bubble_diagram(vinfo, ext_i, ext_j, edges, operator, q0=True)
                        diagrams.append(d)

        return diagrams

    def _close_two_vertex_diagram(
        self, vinfo_a: SMVertexInfo, ext_a: int, vinfo_b: SMVertexInfo, ext_b: int, rem_a: list, rem_b: list, operator: Any
    ) -> list[AdjacencyDiagram]:
        """Connect two vertices via their remaining legs.

        For Type B q≠0: each vertex has one external leg; remaining legs
        form internal propagators connecting the two vertices.
        """
        diagrams = []
        n_a, n_b = len(rem_a), len(rem_b)

        if n_a == 2 and n_b == 2:
            # Each vertex has 2 remaining legs → 2 internal propagators connecting them
            # Try all compatible pairings
            idxs_a = [k for k, _ in rem_a]
            fts_a = [ft for _, ft in rem_a]
            idxs_b = [k for k, _ in rem_b]
            fts_b = [ft for _, ft in rem_b]

            for ia in range(2):
                for ib in range(2):
                    ja = 1 - ia
                    jb = 1 - ib
                    if self._can_form_propagator(fts_a[ia], fts_b[ib]) and self._can_form_propagator(
                        fts_a[ja], fts_b[jb]
                    ):
                        edges = [
                            (idxs_a[ia], idxs_b[ib], fts_a[ia]),
                            (idxs_a[ja], idxs_b[jb], fts_a[ja]),
                        ]
                        d = self._build_two_vertex_diagram(vinfo_a, ext_a, vinfo_b, ext_b, edges, operator, q0=False)
                        diagrams.append(d)

        elif n_a == 1 and n_b == 1:
            # Simple: one internal propagator between them
            (ka, fta) = rem_a[0]
            (kb, ftb) = rem_b[0]
            if self._can_form_propagator(fta, ftb):
                d = self._build_two_vertex_diagram(vinfo_a, ext_a, vinfo_b, ext_b, [(ka, kb, fta)], operator, q0=False)
                diagrams.append(d)

        return diagrams

    def _can_form_propagator(self, ft1: str, ft2: str) -> bool:
        """Check if two field types can form a valid propagator.

        A propagator connects a field and its anti-field:
        - F ↔ -F (fermion propagator)
        - V ↔ -V or V ↔ V (vector, with self-conjugate option)
        - S ↔ -S or S ↔ S (scalar, with self-conjugate option)
        """
        base1 = ft1.lstrip("-")
        base2 = ft2.lstrip("-")
        if base1 != base2:
            return False  # different particle types
        # At least one must be anti (or both self-conjugate)
        is_anti1 = ft1.startswith("-")
        is_anti2 = ft2.startswith("-")
        return is_anti1 != is_anti2  # one is anti, the other is not

    def _build_bubble_diagram(        self,
        vinfo: SMVertexInfo,
        ext_i: int,
        ext_j: int,
        internal_edges: list[tuple[int, int, str]],
        operator: Any,
        q0: bool = True,
    ) -> AdjacencyDiagram:
        """Build a Type A (bubble) adjacency diagram."""
        2 + len(internal_edges)  # 2 external + N internal

        # Vertex 0: CGC operator insertion
        # Vertex 1: SM interaction
        vertices = [
            AdjacencyVertex(vid=0, vtype="CGC_op"),
            AdjacencyVertex(
                vid=1,
                vtype="SM",
                sm_vertex_index=vinfo.index,
                fields_in=vinfo.fields,
                coupling=vinfo.coupling,
            ),
        ]

        # External edges
        edges = [
            AdjacencyEdge(
                eid=0,
                vid_from=0,
                vid_to=1,
                field_type=self._get_field_type(vinfo.fields[ext_i]),
                is_external=True,
                momentum_label="q=0" if q0 else "q_nonzero",
            ),
            AdjacencyEdge(
                eid=1,
                vid_from=0,
                vid_to=1,
                field_type=self._get_field_type(vinfo.fields[ext_j]),
                is_external=True,
                momentum_label="q=0" if q0 else "q_nonzero",
            ),
        ]

        # Internal propagators (loop back to same vertex)
        eid = 2
        for _k1, _k2, ft in internal_edges:
            edges.append(
                AdjacencyEdge(
                    eid=eid,
                    vid_from=1,
                    vid_to=1,
                    field_type=self._get_field_type(ft),
                    is_external=False,
                    is_fast_mode=True,
                )
            )
            eid += 1

        d = AdjacencyDiagram(
            operator_name=operator.name if hasattr(operator, "name") else str(operator),
            loop_order=1,
            vertices=vertices,
            edges=edges,
            external_legs=[0, 1],
        )
        d.compute_id()
        return d

    def _build_two_vertex_diagram(        self,
        vinfo_a: SMVertexInfo,
        ext_a: int,
        vinfo_b: SMVertexInfo,
        ext_b: int,
        internal_edges: list[tuple[int, int, str]],
        operator: Any,
        q0: bool = False,
    ) -> AdjacencyDiagram:
        """Build a Type B (q≠0, two-vertex) adjacency diagram."""
        # Vertex 0: CGC operator insertion
        # Vertex 1: SM interaction A
        # Vertex 2: SM interaction B
        vertices = [
            AdjacencyVertex(vid=0, vtype="CGC_op"),
            AdjacencyVertex(
                vid=1, vtype="SM", sm_vertex_index=vinfo_a.index, fields_in=vinfo_a.fields, coupling=vinfo_a.coupling
            ),
            AdjacencyVertex(
                vid=2, vtype="SM", sm_vertex_index=vinfo_b.index, fields_in=vinfo_b.fields, coupling=vinfo_b.coupling
            ),
        ]

        # External edges
        edges = [
            AdjacencyEdge(
                eid=0,
                vid_from=0,
                vid_to=1,
                field_type=self._get_field_type(vinfo_a.fields[ext_a]),
                is_external=True,
                momentum_label="q_nonzero",
            ),
            AdjacencyEdge(
                eid=1,
                vid_from=0,
                vid_to=2,
                field_type=self._get_field_type(vinfo_b.fields[ext_b]),
                is_external=True,
                momentum_label="q_nonzero",
            ),
        ]

        # Internal propagators
        eid = 2
        for _k1, _k2, ft in internal_edges:
            edges.append(
                AdjacencyEdge(
                    eid=eid,
                    vid_from=1,
                    vid_to=2,
                    field_type=self._get_field_type(ft),
                    is_external=False,
                    is_fast_mode=True,
                )
            )
            eid += 1

        d = AdjacencyDiagram(
            operator_name=operator.name if hasattr(operator, "name") else str(operator),
            loop_order=1,
            vertices=vertices,
            edges=edges,
            external_legs=[0, 1],
        )
        d.compute_id()
        return d

    # ═══════════════════════════════════════════════════════════
    # Two-loop specific topologies (Phase 3)
    # ═══════════════════════════════════════════════════════════

    def enumerate_two_loop_crossed_ladder(self, operator: Any, n_external: int = 2) -> list[AdjacencyDiagram]:
        r"""Generate two-loop crossed-ladder diagrams.

        Topology (CGC notation):
             operator(v0)
              /        \
           ext0        ext1
            /            \
        v1(SM_a)       v3(SM_b)
           |  \        /  |
           |   v2(SM_c)   |
           |   /      \   |
           |  /        \  |
           | /          \ |
        v4(SM_d)       v5(SM_e)
              \        /
             (internal crossing)

        Simplified: 2 CGC_op vertices + 4 SM vertices + 2-loop crossings.
        One diagram per field_type pair → dedup removes redundancies.

        The crossed topology competes with bubble resummation in the
        suppression criterion: the ratio (crossed ladder)/(ladder sum)
        determines whether the resummation series converges.
        """
        diagrams: list[AdjacencyDiagram] = []
        op_name = operator.name if hasattr(operator, "name") else str(operator)
        loop_types = self._loop_field_types(operator)

        for ft in loop_types:
            base = ft.lstrip("-")
            rep_v = self._find_representative_vertex(base, min_legs=1)

            # Crossed ladder: 2 operator vertices + 4 SM vertices
            # Each SM vertex has 3 fields (use representative 3-pt vertex)
            if rep_v is None:
                continue

            # Build the crossed-ladder adjacency graph:
            # v0=CGC_op → ext0, ext1
            # v1=SM_a, v2=SM_b, v3=SM_c, v4=SM_d (4 internal SM vertices)
            # edges form 2 crossed internal loops

            vertices = [
                AdjacencyVertex(vid=0, vtype="CGC_op"),
                AdjacencyVertex(vid=1, vtype="SM", sm_vertex_index=rep_v.index, coupling=rep_v.coupling),
                AdjacencyVertex(vid=2, vtype="SM", sm_vertex_index=rep_v.index, coupling=rep_v.coupling),
                AdjacencyVertex(vid=3, vtype="SM", sm_vertex_index=rep_v.index, coupling=rep_v.coupling),
                AdjacencyVertex(vid=4, vtype="SM", sm_vertex_index=rep_v.index, coupling=rep_v.coupling),
            ]

            edges = [
                # External legs
                AdjacencyEdge(
                    eid=0, vid_from=0, vid_to=1, field_type=base, is_external=True, momentum_label="q_nonzero"
                ),
                AdjacencyEdge(
                    eid=1, vid_from=0, vid_to=3, field_type=base, is_external=True, momentum_label="q_nonzero"
                ),
                # Loop 1: v1 → v2 → v3 (upper arc)
                AdjacencyEdge(eid=2, vid_from=1, vid_to=2, field_type=base, is_external=False, is_fast_mode=True),
                AdjacencyEdge(eid=3, vid_from=2, vid_to=3, field_type=base, is_external=False, is_fast_mode=True),
                # Loop 2: v1 → v4 → v3 (lower arc, crosses upper)
                AdjacencyEdge(eid=4, vid_from=1, vid_to=4, field_type=base, is_external=False, is_fast_mode=True),
                AdjacencyEdge(eid=5, vid_from=4, vid_to=3, field_type=base, is_external=False, is_fast_mode=True),
            ]

            d = AdjacencyDiagram(
                operator_name=f"{op_name}/{ft}-crossed-ladder",
                loop_order=2,
                vertices=vertices,
                edges=edges,
                external_legs=[0, 1],
            )
            d.compute_id()
            diagrams.append(d)

            # q=0 variant: same topology, different external momentum
            verts_q0 = [
                AdjacencyVertex(vid=0, vtype="CGC_op"),
                AdjacencyVertex(vid=1, vtype="SM", sm_vertex_index=rep_v.index, coupling=rep_v.coupling),
                AdjacencyVertex(vid=2, vtype="SM", sm_vertex_index=rep_v.index, coupling=rep_v.coupling),
                AdjacencyVertex(vid=3, vtype="SM", sm_vertex_index=rep_v.index, coupling=rep_v.coupling),
                AdjacencyVertex(vid=4, vtype="SM", sm_vertex_index=rep_v.index, coupling=rep_v.coupling),
            ]
            edges_q0 = [
                AdjacencyEdge(eid=0, vid_from=0, vid_to=1, field_type=base, is_external=True, momentum_label="q=0"),
                AdjacencyEdge(eid=1, vid_from=0, vid_to=3, field_type=base, is_external=True, momentum_label="q=0"),
                AdjacencyEdge(eid=2, vid_from=1, vid_to=2, field_type=base, is_external=False, is_fast_mode=True),
                AdjacencyEdge(eid=3, vid_from=2, vid_to=3, field_type=base, is_external=False, is_fast_mode=True),
                AdjacencyEdge(eid=4, vid_from=1, vid_to=4, field_type=base, is_external=False, is_fast_mode=True),
                AdjacencyEdge(eid=5, vid_from=4, vid_to=3, field_type=base, is_external=False, is_fast_mode=True),
            ]
            d_q0 = AdjacencyDiagram(
                operator_name=f"{op_name}/{ft}-crossed-ladder",
                loop_order=2,
                vertices=verts_q0,
                edges=edges_q0,
                external_legs=[0, 1],
            )
            d_q0.compute_id()
            diagrams.append(d_q0)

        return deduplicate_diagrams(diagrams)

    def enumerate_two_loop_vertex_correction(self, operator: Any) -> list[AdjacencyDiagram]:
        """Generate two-loop vertex-correction diagrams.

        Topology: one-loop bubble where one SM vertex has an
        additional self-energy bubble insertion (vertex dressing).

        This contributes to the two-loop correction of V_eff and
        modifies the critical coupling λ_crit at next-to-leading order.

              operator
              /      \
           ext0      ext1
            /          \
        v1(SM) ← internal_loop → v2(SM)
           |                       |
           └── bubble_insertion ───┘
        """
        diagrams: list[AdjacencyDiagram] = []
        op_name = operator.name if hasattr(operator, "name") else str(operator)
        loop_types = self._loop_field_types(operator)

        for ft in loop_types:
            base = ft.lstrip("-")
            rep_v = self._find_representative_vertex(base, min_legs=1)

            if rep_v is None:
                continue

            # Vertex correction topology:
            # v0=CGC_op, v1=SM_a, v2=SM_b (the vertex-dressed one), v3=SM_c (bubble insertion)
            vertices = [
                AdjacencyVertex(vid=0, vtype="CGC_op"),
                AdjacencyVertex(vid=1, vtype="SM", sm_vertex_index=rep_v.index, coupling=rep_v.coupling),
                AdjacencyVertex(vid=2, vtype="SM", sm_vertex_index=rep_v.index, coupling=rep_v.coupling),
                AdjacencyVertex(vid=3, vtype="SM", sm_vertex_index=rep_v.index, coupling=rep_v.coupling),
            ]

            edges = [
                # External legs
                AdjacencyEdge(eid=0, vid_from=0, vid_to=1, field_type=base, is_external=True, momentum_label="q=0"),
                AdjacencyEdge(eid=1, vid_from=0, vid_to=2, field_type=base, is_external=True, momentum_label="q=0"),
                # Main internal loop (v1 ↔ v2)
                AdjacencyEdge(eid=2, vid_from=1, vid_to=2, field_type=base, is_external=False, is_fast_mode=True),
                # Bubble insertion on v2 (v2 → v3 → v2)
                AdjacencyEdge(eid=3, vid_from=2, vid_to=3, field_type=base, is_external=False, is_fast_mode=True),
                AdjacencyEdge(eid=4, vid_from=3, vid_to=2, field_type=base, is_external=False, is_fast_mode=True),
            ]

            d = AdjacencyDiagram(
                operator_name=f"{op_name}/{ft}-vertex-corr",
                loop_order=2,
                vertices=vertices,
                edges=edges,
                external_legs=[0, 1],
            )
            d.compute_id()
            diagrams.append(d)

        return deduplicate_diagrams(diagrams)


# ═══════════════════════════════════════════════════════════════
# 4. Integration helpers
# ═══════════════════════════════════════════════════════════════


def adjacency_to_cgc_diagram(adj: AdjacencyDiagram) -> Diagram:
    """Convert adjacency diagram to existing CGC Diagram format.

    Maps:
      - AdjacencyEdge.is_external=True  → external_lines
      - AdjacencyEdge.is_external=False → internal_lines
      - AdjacencyVertex → Vertex (fields, coupling, momentum_routing)
    """
    from .diagram_generator import Diagram, Vertex

    # Build vertices
    vertices = []
    for av in adj.vertices:
        fields = []
        momentum = {}
        # Collect field labels from edges connected to this vertex
        connected_edges = [e for e in adj.edges if e.vid_from == av.vid or e.vid_to == av.vid]
        for e in connected_edges:
            label = f"{e.field_type}_e{e.eid}"
            if e.is_external:
                label += "_ext"
            fields.append(label)
            momentum[label] = e.momentum_label if e.momentum_label else ""
        vertices.append(
            Vertex(
                fields=fields,
                coupling=av.coupling or (f"{adj.operator_name}_insertion" if av.vtype == "CGC_op" else "SM_vertex"),
                momentum_routing=momentum,
            )
        )

    # Build internal/external lines
    internal_lines = []
    external_lines = []
    for e in adj.edges:
        label = e.momentum_label if e.momentum_label else "loop"
        if e.is_external:
            external_lines.append(label)
        else:
            internal_lines.append((e.field_type, label))

    # Determine topology label
    q0 = all(ml == "q=0" for ml in external_lines) if external_lines else False
    topology_label = "bubble" if q0 else "nonzero_q"

    # Determine momentum_transfer
    momentum_transfer = "0" if q0 else "q"

    return Diagram(
        id=adj.diagram_id or adj.compute_id(),
        loop_number=adj.loop_order,
        momentum_transfer=momentum_transfer,
        topology_label=topology_label,
        n_bubbles=1 if q0 else 0,
        n_irreducible_insertions=0,
        has_line_crossing=False,
        has_vertex_dressing=False,
        vertices=vertices,
        internal_lines=internal_lines,
        external_lines=external_lines,
        description=f"One-loop {adj.operator_name} diagram ({topology_label})",
    )


if __name__ == "__main__":
    # Quick smoke test
    enumerator = DiagramEnumerator()
    print(f"Loaded {len(enumerator.physical_vertices)} physical SM vertices")
    print(f"Field type index: {list(enumerator.by_field_type.keys())}")
