"""
Tests: Conservation Checker (Phase 2.4)

Verifies that conservation-law protection rules are correctly applied.
This is the CORE of the CGC method — the conservation-law logic that
replaces the perturbative a₂ coefficient.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgc.engine.diagram_generator import OperatorSpec, OperatorType
from cgc.engine.topology_classifier import (
    TopologyClassification, TopologyLabel, TopologyClass,
)
from cgc.engine.conservation_checker import (
    ConservationChecker, ProtectionBasis, ProtectionVerdict,
)


@pytest.fixture
def checker():
    return ConservationChecker()


@pytest.fixture
def empty_topo():
    return TopologyClassification(operator_name="test")


# ── Operator Fixtures ───────────────────────────────────────────────────

@pytest.fixture
def tmunu_operator():
    return OperatorSpec(
        "Tμν", OperatorType.CONSERVED_CURRENT,
        2, 2, 2, 4, is_protected=True,
        protection_source="Ward identity",
    )


@pytest.fixture
def gauge_operator():
    return OperatorSpec(
        "Fμν^a", OperatorType.GAUGE_FIELD_STRENGTH,
        2, 1, 2, 4, is_protected=True,
        protection_source="BRST symmetry",
    )


@pytest.fixture
def scalar_operator():
    return OperatorSpec(
        "φ†φ", OperatorType.UNPROTECTED_SCALAR,
        0, 0, 2, 2, is_protected=False,
        protection_source="None",
    )


@pytest.fixture
def fermion_operator():
    return OperatorSpec(
        "ψ̄ψ", OperatorType.UNPROTECTED_FERMION,
        0, 0, 2, 3, is_protected=False,
        protection_source="None",
    )


# ── Unit Tests ──────────────────────────────────────────────────────────

class TestConservationChecker:
    """Conservation-law protection logic tests."""

    def test_checker_creation(self, checker):
        assert checker is not None
        assert len(checker.PROTECTION_RULES) >= 4

    def test_tmunu_is_protected(self, checker, tmunu_operator, empty_topo):
        report = checker.check(tmunu_operator, empty_topo)
        assert report.verdict.is_protected is True
        assert report.verdict.protection_basis == ProtectionBasis.WARD_IDENTITY
        assert report.verdict.matrix_element_nonzero is True

    def test_gauge_is_protected(self, checker, gauge_operator, empty_topo):
        report = checker.check(gauge_operator, empty_topo)
        assert report.verdict.is_protected is True
        assert report.verdict.protection_basis == ProtectionBasis.BRST_SYMMETRY
        assert report.verdict.matrix_element_nonzero is True

    def test_scalar_is_unprotected(self, checker, scalar_operator, empty_topo):
        report = checker.check(scalar_operator, empty_topo)
        assert report.verdict.is_protected is False
        assert report.verdict.protection_basis == ProtectionBasis.NONE
        assert report.verdict.matrix_element_nonzero is False

    def test_fermion_is_unprotected(self, checker, fermion_operator, empty_topo):
        report = checker.check(fermion_operator, empty_topo)
        assert report.verdict.is_protected is False
        assert report.verdict.matrix_element_nonzero is False

    def test_injection_nonzero_for_protected(self, checker, tmunu_operator, empty_topo):
        """Protected operators → injection guaranteed nonzero."""
        report = checker.check(tmunu_operator, empty_topo)
        assert checker.is_injection_nonzero(report) is True

    def test_injection_may_be_zero_for_unprotected(self, checker, scalar_operator, empty_topo):
        """Unprotected operators → no conservation guarantee."""
        report = checker.check(scalar_operator, empty_topo)
        assert checker.is_injection_nonzero(report) is False

    def test_unknown_operator_type(self, checker, empty_topo):
        """Operators with type OTHER get no protection."""
        unknown = OperatorSpec(
            "unknown", OperatorType.OTHER,
            0, 0, 2, 3, is_protected=False, protection_source="",
        )
        report = checker.check(unknown, empty_topo)
        assert report.verdict.is_protected is False

    def test_report_summary(self, checker, tmunu_operator, empty_topo):
        """Report summary should contain key fields."""
        report = checker.check(tmunu_operator, empty_topo)
        summary = report.summary()
        assert "Ward" in summary or "WARD_IDENTITY" in summary
        assert "True" in summary


@pytest.mark.physics
class TestConservationCheckerPhysics:
    """Physics benchmarks for conservation-law logic."""

    def test_ward_identity_does_not_force_vanishing(self, checker, tmunu_operator, empty_topo):
        """
        Physics benchmark: Ward identity q_μ M^{μν...} = 0 does NOT force
        M = 0 at q=0. The identity becomes vacuous (0 = 0) and the matrix
        element is generically nonzero.

        This is the core physics insight of the CGC method:
        conservation laws guarantee injection term nonzeroness
        without any perturbative assumption.
        """
        report = checker.check(tmunu_operator, empty_topo)
        # The verdict MUST confirm matrix_element_nonzero
        assert report.verdict.matrix_element_nonzero is True
        # The theorem reference must include Ward
        assert "Ward" in report.verdict.theorem_reference

    def test_all_protection_rules_consistent(self, checker):
        """All protection rules must be internally consistent."""
        for op_type, verdict in checker.PROTECTION_RULES.items():
            assert isinstance(verdict, ProtectionVerdict)
            assert isinstance(verdict.protection_basis, ProtectionBasis)
            # Protected → matrix elements guaranteed nonzero
            if verdict.is_protected:
                assert verdict.matrix_element_nonzero is True
            # Unprotected → no guarantee
            else:
                assert verdict.matrix_element_nonzero is False

    def test_cgc_upgrade_over_heat_kernel(self, checker, tmunu_operator, empty_topo):
        """
        The CGC method's central upgrade over the heat-kernel approach:
        
        Heat kernel: Δ_σ ≠ 0 relies on a₂ ≠ 0 (perturbative Seeley-DeWitt
        coefficient). Could be canceled by nonperturbative effects.
        
        CGC: Δ_σ ≠ 0 relies on Ward/BRST identity → algebraic guarantee
        that matrix elements at q=0 are nonzero. No perturbative assumption.
        
        This test verifies the CGC engine correctly applies this logic.
        """
        # For Tμν (Ward-protected): injection IS guaranteed nonzero
        report = checker.check(tmunu_operator, empty_topo)
        assert checker.is_injection_nonzero(report) is True
        
        # For scalar φ†φ (unprotected): NO guarantee
        scalar_op = OperatorSpec(
            "φ†φ", OperatorType.UNPROTECTED_SCALAR,
            0, 0, 2, 2, is_protected=False,
        )
        scalar_report = checker.check(scalar_op, empty_topo)
        assert checker.is_injection_nonzero(scalar_report) is False
