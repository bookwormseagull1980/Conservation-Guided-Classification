"""
Diagram Generator — Phase 2.1
============================================================================

Generates all connected irreducible Feynman diagrams up to a given loop order
for a specified composite operator, using Qgraf or a built-in topological generator.

Input:  composite operator + coarse-graining scheme
Output: list of diagrams with topological metadata

Architecture:
  OperatorSpec  — abstract definition of the composite operator
  Diagram       — single diagram with momentum routing info
  DiagramSet    — collection of diagrams with completeness verification

Supported generators:
  - qgraf:  call external Qgraf binary (preferred for full SM)
  - feynarts: call Mathematica/FeynArts
  - builtin: topological brute-force for single-loop (self-contained fallback)

Verification target (Phase 2.6 benchmark):
  For Tμν spin-2 channel at one-loop, output must match Appendix E Figure 1.
"""


# References
#     FRG one-loop expansion: Wetterich (1993), Phys. Lett. B 301, 90
#     Diagram topology enumeration: standard perturbation theory
#

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

# ── Data Structures ──────────────────────────────────────────────────────


class OperatorType(Enum):
    """Classification of the composite operator."""

    CONSERVED_CURRENT = auto()  # e.g. Tμν (Ward identity), Jμ (Noether)
    GAUGE_FIELD_STRENGTH = auto()  # e.g. Fμν^a (BRST protected)
    UNPROTECTED_SCALAR = auto()  # e.g. φ†φ, Higgs quartic
    UNPROTECTED_FERMION = auto()  # e.g. ψ̄ψ, ψ̄γ5ψ
    OTHER = auto()


class GeneratorBackend(Enum):
    QGRAF = "qgraf"
    FEYNARTS = "feynarts"
    BUILTIN = "builtin"


@dataclass
class OperatorSpec:
    """Abstract definition of a composite operator."""

    name: str
    op_type: OperatorType
    lorentz_rank: int  # 0=scalar, 1=vector, 2=tensor
    spin_channel: int  # 0, 1, 2
    external_momenta: int  # number of external legs
    mass_dimension: int  # mass dimension of the operator
    is_protected: bool  # conservation-law protected?
    protection_source: str = ""  # "Ward", "BRST", "None"


@dataclass
class Vertex:
    """A single interaction vertex in a diagram."""

    fields: list[str]
    coupling: str  # symbolic coupling constant
    momentum_routing: dict[str, str] = field(default_factory=dict)


@dataclass
class Diagram:
    """
    Single Feynman diagram with topological and momentum metadata.

    Attributes:
        id: unique diagram identifier
        vertices: list of interaction vertices
        internal_lines: list of (field_type, momentum_label) pairs
        external_lines: list of external leg momenta
        loop_number: number of loops
        is_one_particle_irreducible: 1PI flag
        is_connected: connectivity flag
        momentum_transfer: q vector from fast to slow modes
        topology_label: "bubble", "ladder", "crossed_ladder", "vertex_correction", etc.
    """

    id: str
    vertices: list[Vertex] = field(default_factory=list)
    internal_lines: list[tuple[str, str]] = field(default_factory=list)
    external_lines: list[str] = field(default_factory=list)
    loop_number: int = 0
    is_one_particle_irreducible: bool = True
    is_connected: bool = True
    momentum_transfer: str | None = None
    topology_label: str = ""
    # Explicit topology metadata (set by generator, used by topology classifier)
    n_bubbles: int = 0
    n_irreducible_insertions: int = 0
    has_line_crossing: bool = False
    has_vertex_dressing: bool = False
    # Human-readable description for auditability
    description: str = ""

    def __repr__(self) -> str:
        return f"Diagram({self.id}, L={self.loop_number}, topo={self.topology_label})"


@dataclass
class DiagramSet:
    """Collection of diagrams with completeness metadata."""

    diagrams: list[Diagram]
    operator: OperatorSpec
    max_loop_order: int
    total_count: int
    generator_backend: GeneratorBackend
    is_complete: bool = False  # verified against known enumeration

    @property
    def one_loop_diagrams(self) -> list[Diagram]:
        return [d for d in self.diagrams if d.loop_number == 1]

    @property
    def by_topology(self) -> dict[str, list[Diagram]]:
        groups: dict[str, list[Diagram]] = {}
        for d in self.diagrams:
            groups.setdefault(d.topology_label, []).append(d)
        return groups


# ── Generator Interface ──────────────────────────────────────────────────


class DiagramGenerator:
    """
    Produces the complete set of connected, 1PI Feynman diagrams for a given
    composite operator at specified loop order.

    Backends:
      - BUILTIN: first-principles one-loop enumeration using
                 cgc.engine.one_loop_generator. For each SM field that
                 couples to the operator, generates q=0 bubble + q≠0
                 variants. Multi-loop ladders are built by resummation.
      - QGRAF:   external binary call (for multi-loop / full SM)
      - FEYNARTS: Mathematica-based generation

    Current capability:
      - All operator types at one-loop: enumeration from SM field content ✅
      - Multi-loop: builtin multi-loop generator (multi_loop_generator.py)
        for sunset/double-bubble/figure-8 topologies; QGRAF backend as
        an alternative external option
    """

    # Expected one-loop diagram counts derived from SM field coupling rules.
    # These are INDEPENDENT of the generator output — they are computed
    # from the same coupling rules in one_loop_generator.expected_one_loop_count().
    # The _verify_completeness method checks generator output against these.

    def __init__(self, backend: GeneratorBackend = GeneratorBackend.BUILTIN):
        self.backend = backend
        self._qgraf_path: str | None = None

    # ── Public API ──

    def generate(self, operator: OperatorSpec, max_loops: int = 1) -> DiagramSet:
        """
        Generate all diagrams for the operator up to max_loops.

        Args:
            operator: composite operator specification
            max_loops: maximum loop order (default 1 for Phase 1)

        Returns:
            DiagramSet with all generated diagrams and metadata
        """
        if self.backend == GeneratorBackend.BUILTIN:
            diagrams = self._generate_builtin(operator, max_loops)
        elif self.backend == GeneratorBackend.QGRAF:
            diagrams = self._generate_qgraf(operator, max_loops)
        elif self.backend == GeneratorBackend.FEYNARTS:
            diagrams = self._generate_feynarts(operator, max_loops)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        diag_set = DiagramSet(
            diagrams=diagrams,
            operator=operator,
            max_loop_order=max_loops,
            total_count=len(diagrams),
            generator_backend=self.backend,
        )
        diag_set.is_complete = self._verify_completeness(diag_set)
        return diag_set

    # ── Backend Implementations ──

    def _generate_builtin(self, operator: OperatorSpec, max_loops: int) -> list[Diagram]:
        """
        Builtin diagram generation.

        Strategy:
          - max_loops == 1: use one_loop_generator (first-principles)
          - max_loops >= 2: delegate to QGRAF backend (auto-detect binary);
            fall back to NotImplementedError if QGRAF unavailable.
        """
        diagrams: list[Diagram] = []

        if max_loops >= 1:
            diagrams.extend(self._generate_one_loop(operator))

        if max_loops >= 2:
            # Try QGRAF first, fall back to builtin multi-loop
            try:
                multi = self._generate_qgraf(operator, max_loops)
                existing_ids = {d.id for d in diagrams}
                for d in multi:
                    if d.id not in existing_ids:
                        diagrams.append(d)
            except RuntimeError:
                # QGRAF unavailable — use builtin multi-loop generator
                import importlib

                mod = importlib.import_module("cgc.engine.multi_loop_generator")
                multi = mod.generate_multi_loop_diagrams(operator, max_loops)
                existing_ids = {d.id for d in diagrams}
                for d in multi:
                    if d.id not in existing_ids:
                        diagrams.append(d)

        return diagrams

    def _generate_one_loop(self, operator: OperatorSpec) -> list[Diagram]:
        """
        Produce all one-loop 1PI diagrams for the given operator.

        Uses one_loop_generator for first-principles enumeration:
          - Determines which SM fields couple to the operator
          - For each coupled field: q=0 bubble + q≠0 variant
          - Automatically sets correct topology metadata

        Multi-loop ladders are NOT generated here — they are
        constructed by the resummation module from Π₀.
        """
        import importlib

        mod = importlib.import_module("cgc.engine.one_loop_generator")
        return mod.generate_one_loop_diagrams(operator)  # type: ignore[no-any-return]

    def _generate_qgraf(self, operator: OperatorSpec, max_loops: int) -> list[Diagram]:
        """Generate diagrams via external QGRAF binary.

        Fully integrated: generates model file, runs QGRAF, parses output,
        converts to CGC Diagram objects with correct topology metadata.

        Falls back to one_loop_generator if QGRAF binary is unavailable
        and max_loops == 1.
        """
        import importlib

        mod = importlib.import_module("cgc.engine.qgraf_backend")

        diag_dicts, log = mod.run_qgraf(
            operator_name=operator.op_type.name,
            max_loops=max_loops,
        )

        if not diag_dicts:
            # QGRAF unavailable or failed — fall back to builtin for one-loop
            if max_loops == 1:
                print(f"[qgraf] {log} — falling back to builtin one-loop generator")
                return self._generate_one_loop(operator)
            raise RuntimeError(f"QGRAF unavailable and multi-loop (L={max_loops}) builtin not implemented. {log}")

        # Convert dicts to Diagram objects
        diagrams = [Diagram(**d) for d in diag_dicts]
        print(f"[qgraf] {log}")
        return diagrams

    def _generate_feynarts(self, operator: OperatorSpec, max_loops: int) -> list[Diagram]:
        """Generate diagrams via Mathematica/FeynArts.

        Dual-mode: calls Mathematica if available, otherwise
        parses FeynArts SM.mod for combinatorial enumeration.
        """
        from .feynarts_backend import FeynArtsBackend

        backend = FeynArtsBackend()
        diag_dicts, log = backend.generate(
            operator_name=operator.op_type.name,
            max_loops=max_loops,
        )

        if not diag_dicts:
            raise RuntimeError(f"FeynArts backend produced no diagrams. {log}")

        diagrams = [Diagram(**d) for d in diag_dicts]
        print(f"[feynarts] {log}")
        return diagrams

    # ── Completeness Verification ──

    def _verify_completeness(self, diag_set: DiagramSet) -> bool:
        """
        Verify diagram set completeness against independent expectation.

        Uses expected_one_loop_count() from one_loop_generator, which
        computes the expected count from the SAME coupling rules as the
        generator — but through a separate code path. This is NOT circular:
        it's a consistency check that the generator correctly enumerates
        all field-type × kinematic-class combinations.

        Returns:
            True if count matches expectation,
            False if count mismatch or operator type unrecognized.
        """
        import importlib

        mod = importlib.import_module("cgc.engine.one_loop_generator")
        expected = mod.expected_one_loop_count(diag_set.operator)
        if expected > 0:
            return diag_set.total_count == expected  # type: ignore[no-any-return]
        return False
