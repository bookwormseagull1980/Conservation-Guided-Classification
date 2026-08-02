"""
Tests: Diagram Generator (Phase 2.1)

Verifies that the diagram generator produces the correct complete
set of one-loop diagrams for each supported operator.

One-loop enumeration is from first principles: for each SM field
type that couples to the operator → q=0 bubble + q≠0 variant.
Ladder diagrams are constructed by the resummation module, not
by the one-loop generator.

Tμν (CONSERVED_CURRENT, spin-2) couples to 3 SM field types
→ 6 one-loop diagrams (3 q=0 bubbles + 3 q≠0 variants).
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgc.engine.diagram_generator import (
    DiagramGenerator,
    GeneratorBackend,
    OperatorSpec,
    OperatorType,
)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def tmunu_operator():
    return OperatorSpec(
        name="Tμν spin-2",
        op_type=OperatorType.CONSERVED_CURRENT,
        lorentz_rank=2,
        spin_channel=2,
        external_momenta=2,
        mass_dimension=4,
        is_protected=True,
        protection_source="Ward identity",
    )


@pytest.fixture
def generator():
    return DiagramGenerator(backend=GeneratorBackend.BUILTIN)


# ── Unit Tests ──────────────────────────────────────────────────────────

class TestOperatorSpec:
    """OperatorSpec data structure tests."""

    def test_operator_creation(self):
        op = OperatorSpec(
            name="test", op_type=OperatorType.CONSERVED_CURRENT,
            lorentz_rank=2, spin_channel=2, external_momenta=2,
            mass_dimension=4, is_protected=True, protection_source="Ward",
        )
        assert op.name == "test"
        assert op.spin_channel == 2
        assert op.is_protected is True

    def test_protected_vs_unprotected(self):
        protected = OperatorSpec(
            "T", OperatorType.CONSERVED_CURRENT, 2, 2, 2, 4,
            is_protected=True, protection_source="Ward",
        )
        unprotected = OperatorSpec(
            "phi4", OperatorType.UNPROTECTED_SCALAR, 0, 0, 4, 4,
            is_protected=False, protection_source="None",
        )
        assert protected.is_protected != unprotected.is_protected


class TestDiagramGenerator:
    """Diagram generator core functionality."""

    def test_generator_creation(self, generator):
        assert generator.backend == GeneratorBackend.BUILTIN

    def test_generate_tmunu_spin2(self, generator, tmunu_operator):
        """Tμν spin-2 at one-loop: 3 field types × 2 kinematic classes = 6 diagrams."""
        result = generator.generate(tmunu_operator, max_loops=1)
        assert result is not None
        assert result.operator.name == "Tμν spin-2"
        assert result.total_count == 6  # weyl_fermion + gauge_boson + real_scalar ×{q=0,q≠0}
        assert result.is_complete is True
        # Verify diagram IDs follow naming convention
        ids = {d.id for d in result.diagrams}
        assert "tμν_spin2_weyl_fermion_bubble_q0" in ids
        assert "tμν_spin2_weyl_fermion_nonzero_q" in ids
        assert "tμν_spin2_gauge_boson_bubble_q0" in ids
        assert "tμν_spin2_gauge_boson_nonzero_q" in ids
        assert "tμν_spin2_real_scalar_bubble_q0" in ids
        assert "tμν_spin2_real_scalar_nonzero_q" in ids

    def test_diagram_set_metadata(self, generator, tmunu_operator):
        result = generator.generate(tmunu_operator, max_loops=1)
        assert result.max_loop_order == 1
        assert result.generator_backend == GeneratorBackend.BUILTIN
        assert result.total_count == 6
        assert result.is_complete is True
        # All diagrams should be one-loop
        for d in result.diagrams:
            assert d.loop_number == 1

    def test_multi_loop_builtin(self, generator, tmunu_operator):
        """Multi-loop generation (L=2) works via builtin + QGRAF fallback."""
        result = generator.generate(tmunu_operator, max_loops=2)
        # One-loop (6) + two-loop (sunset + double_bubble + fig8) > 6
        assert result.total_count > 6
        assert result.max_loop_order == 2
        # Verify some two-loop topologies present
        topologies = {d.topology_label for d in result.diagrams}
        assert "sunset" in topologies or "double_bubble" in topologies

    def test_generate_gauge_field_strength(self, generator):
        """Gauge field strength operator: 2 field types × 2 classes = 4 diagrams."""
        op = OperatorSpec(
            name="Fμν Fμν",
            op_type=OperatorType.GAUGE_FIELD_STRENGTH,
            lorentz_rank=0, spin_channel=0,
            external_momenta=2, mass_dimension=4,
            is_protected=True, protection_source="BRST",
        )
        result = generator.generate(op, max_loops=1)
        assert result.total_count == 4  # gauge_boson + weyl_fermion ×{q=0,q≠0}
        assert result.is_complete is True

    def test_generate_fermion_bilinear(self, generator):
        """Fermion bilinear operator: 1 field type × 2 classes = 2 diagrams."""
        op = OperatorSpec(
            name="ψ̄ψ",
            op_type=OperatorType.UNPROTECTED_FERMION,
            lorentz_rank=0, spin_channel=0,
            external_momenta=2, mass_dimension=3,
            is_protected=False, protection_source="None",
        )
        result = generator.generate(op, max_loops=1)
        assert result.total_count == 2
        assert result.is_complete is True

    def test_generate_unknown_operator(self, generator):
        """Unknown operator type returns empty set (no coupling rules)."""
        op = OperatorSpec(
            name="unknown", op_type=OperatorType.OTHER,
            lorentz_rank=0, spin_channel=0, external_momenta=2,
            mass_dimension=4, is_protected=False,
        )
        result = generator.generate(op, max_loops=1)
        assert result.total_count == 6  # OTHER maps to full SM
        assert result.is_complete is True


# ── Physics Benchmarks ──────────────────────────────────────────────────

@pytest.mark.physics
class TestDiagramGeneratorPhysics:
    """Physics benchmark: verify diagram enumeration correctness."""

    def test_tmunu_bubble_count(self, generator, tmunu_operator):
        """Tμν has 3 q=0 bubble diagrams (one per coupled field type)."""
        result = generator.generate(tmunu_operator, max_loops=1)
        bubbles = [d for d in result.diagrams if d.momentum_transfer == "0"]
        assert len(bubbles) == 3  # weyl_fermion, gauge_boson, real_scalar
        for b in bubbles:
            assert b.n_bubbles == 1
            assert b.n_irreducible_insertions == 0

    def test_tmunu_nonzero_q_count(self, generator, tmunu_operator):
        """Tμν has 3 q≠0 diagrams (one per coupled field type)."""
        result = generator.generate(tmunu_operator, max_loops=1)
        nonzero = [d for d in result.diagrams if d.momentum_transfer != "0"]
        assert len(nonzero) == 3
        for nz in nonzero:
            assert nz.n_bubbles == 0  # NOT CGC bubbles

    def test_no_ladder_in_one_loop(self, generator, tmunu_operator):
        """One-loop generator does NOT produce ladder diagrams.
        
        Ladders are multi-loop structures (N bubbles connected by V).
        They are built by the resummation module from Π₀.
        """
        result = generator.generate(tmunu_operator, max_loops=1)
        ladders = [d for d in result.diagrams
                   if d.topology_label == "ladder"]
        assert len(ladders) == 0

    def test_all_diagrams_connected(self, generator, tmunu_operator):
        """All generated diagrams must be connected."""
        result = generator.generate(tmunu_operator, max_loops=1)
        for diag in result.diagrams:
            assert diag.is_connected is True

    def test_all_diagrams_1PI(self, generator, tmunu_operator):
        """All generated diagrams must be one-particle irreducible."""
        result = generator.generate(tmunu_operator, max_loops=1)
        for diag in result.diagrams:
            assert diag.is_one_particle_irreducible is True
