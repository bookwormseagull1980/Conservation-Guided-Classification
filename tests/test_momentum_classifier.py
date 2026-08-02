"""
Tests: Momentum Transfer Classifier (Phase 2.2)

Verifies the q=0 vs q≠0 classification logic.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgc.engine.diagram_generator import (
    Diagram, DiagramSet, OperatorSpec, OperatorType, Vertex,
)
from cgc.engine.momentum_classifier import (
    MomentumClassifier, MomentumClass, MomentumClassification,
)


@pytest.fixture
def classifier():
    return MomentumClassifier()


@pytest.fixture
def empty_diagram_set():
    return DiagramSet(
        diagrams=[],
        operator=OperatorSpec("test", OperatorType.CONSERVED_CURRENT, 2, 2, 2, 4, True),
        max_loop_order=1,
        total_count=0,
        generator_backend="builtin",
    )


@pytest.fixture
def single_bubble_diagram():
    return Diagram(
        id="bubble_1",
        loop_number=1,
        topology_label="bubble",
        momentum_transfer="0",
    )


@pytest.fixture
def ladder_diagram():
    return Diagram(
        id="ladder_2rung",
        loop_number=1,
        topology_label="ladder",
        momentum_transfer="0",  # Correct physics: fast k and -k back-to-back → q=0
    )


@pytest.fixture
def ladder_diagram_no_mt():
    """Ladder diagram WITHOUT explicit momentum_transfer (tests fallback)."""
    return Diagram(
        id="ladder_2rung_no_mt",
        loop_number=1,
        topology_label="ladder",
        momentum_transfer=None,
    )


class TestMomentumClassifier:
    """Unit tests for momentum transfer classification."""

    def test_classifier_creation(self, classifier):
        assert classifier is not None

    def test_empty_diagram_set(self, classifier, empty_diagram_set):
        result = classifier.classify(empty_diagram_set)
        assert result.total == 0

    def test_single_bubble_is_q0(self, classifier, single_bubble_diagram):
        """Single bubble: external legs back-to-back → q=0."""
        result = classifier._classify_one(single_bubble_diagram, None)
        assert result.momentum_class == MomentumClass.ZERO_TRANSFER
        assert result.q_label == "0"

    def test_ladder_is_q0(self, classifier, ladder_diagram):
        """Ladder with explicit momentum_transfer="0": q=0."""
        result = classifier._classify_one(ladder_diagram, None)
        assert result.momentum_class == MomentumClass.ZERO_TRANSFER
        assert result.q_label == "0"

    def test_ladder_fallback_is_q0(self, classifier, ladder_diagram_no_mt):
        """Ladder without explicit momentum_transfer: fallback → q=0."""
        result = classifier._classify_one(ladder_diagram_no_mt, None)
        assert result.momentum_class == MomentumClass.ZERO_TRANSFER
        assert result.q_label == "0"


class TestSuppressionFactor:
    """Oscillatory factor suppression tests."""

    def test_large_sigma_suppression(self, classifier):
        """For σ|q| ≫ 1, suppression is exponentially small."""
        factor = classifier.suppression_strength(q_magnitude=10.0, sigma=10.0)
        # exp(-5000) ≈ 0
        assert factor < 1e-100

    def test_small_sigma_no_suppression(self, classifier):
        """For σ|q| ≪ 1, suppression is weak (near-UV regime)."""
        factor = classifier.suppression_strength(q_magnitude=0.01, sigma=1.0)
        # exp(-5e-5) ≈ 0.99995
        assert factor > 0.999

    def test_crossover_at_sigma_q_equals_one(self, classifier):
        """The crossover σ|q| = 1 marks the boundary."""
        factor = classifier.suppression_strength(q_magnitude=1.0, sigma=1.0)
        # exp(-0.5) ≈ 0.6065
        assert 0.60 < factor < 0.61


@pytest.mark.physics
class TestMomentumClassifierPhysics:
    """Physics benchmarks for momentum transfer classification."""

    def test_oscillatory_suppression_is_exponential(self, classifier):
        """
        Benchmark: verify that suppression is exponential in (σ|q|)².
        This is a mathematical fact from Fourier analysis — not a parameter.
        """
        import math
        q_vals = [0.5, 1.0, 2.0, 4.0, 8.0]
        sigma = 5.0
        for q in q_vals:
            factor = classifier.suppression_strength(q, sigma)
            expected = math.exp(-0.5 * (sigma * q) ** 2)
            assert abs(factor - expected) < 1e-15
