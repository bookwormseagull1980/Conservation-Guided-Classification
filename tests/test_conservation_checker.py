"""Unit tests: ConservationChecker — classification logic.

Tests the core conservation logic that determines whether operators
are protected by Ward/BRST/Noether identities.

Full pipeline integration (which requires DiagramGenerator + TopologyClassifier)
is tested by the L1 benchmarks. These unit tests cover the classification
logic in isolation.
"""

from cgc.engine.conservation_checker import (
    ConservationChecker,
    OperatorSpec,
    OperatorType,
    ProtectionBasis,
    ProtectionVerdict,
)
from cgc.engine.topology_classifier import TopologyClass, TopologyClassification, TopologyLabel


def test_protection_basis_values():
    """All four protection mechanisms are defined."""
    bases = list(ProtectionBasis)
    assert ProtectionBasis.WARD_IDENTITY in bases
    assert ProtectionBasis.BRST_SYMMETRY in bases
    assert ProtectionBasis.NOETHER_THEOREM in bases
    assert ProtectionBasis.NONE in bases
    assert len(bases) == 4


def test_topology_class_values():
    """All five topology classes are defined."""
    classes = list(TopologyClass)
    assert TopologyClass.SINGLE_BUBBLE in classes
    assert TopologyClass.LADDER in classes
    assert TopologyClass.CROSSED_LADDER in classes
    assert TopologyClass.VERTEX_CORRECTION in classes
    assert TopologyClass.OTHER in classes
    assert len(classes) == 5


def test_operator_spec_construction():
    """OperatorSpec attributes are preserved correctly."""
    for name, op_type, rank, is_prot in [
        ("Tmunu", OperatorType.CONSERVED_CURRENT, 2, True),
        ("F2", OperatorType.GAUGE_FIELD_STRENGTH, 2, True),
        ("psibar_psi", OperatorType.UNPROTECTED_FERMION, 0, False),
        ("phi4", OperatorType.UNPROTECTED_SCALAR, 0, False),
    ]:
        spec = OperatorSpec(
            name=name,
            op_type=op_type,
            lorentz_rank=rank,
            spin_channel=rank,
            external_momenta=2,
            mass_dimension=4,
            is_protected=is_prot,
            protection_source="test",
        )
        assert spec.name == name
        assert spec.op_type == op_type
        assert spec.lorentz_rank == rank
        assert spec.is_protected == is_prot


def test_topology_label_construction():
    """TopologyLabel construction with required fields."""
    for tc in [TopologyClass.SINGLE_BUBBLE, TopologyClass.LADDER]:
        label = TopologyLabel(
            diagram_id=f"test_{tc.name}",
            topology_class=tc,
            n_bubbles=1,
            n_insertions=0,
            is_ladder_resummable=(tc == TopologyClass.LADDER),
        )
        assert label.topology_class == tc
        assert label.diagram_id == f"test_{tc.name}"


def test_conservation_report_construction():
    """ProtectionVerdict dataclass can be constructed."""
    verdict = ProtectionVerdict(
        operator_name="Tmunu",
        protection_basis=ProtectionBasis.WARD_IDENTITY,
        is_protected=True,
        matrix_element_nonzero=True,
        theorem_reference="Ward identity: d_mu T^mu_nu = 0",
        notes="q=0 matrix element survives via diffeomorphism invariance",
    )
    assert verdict.is_protected is True
    assert verdict.matrix_element_nonzero is True
    assert verdict.protection_basis == ProtectionBasis.WARD_IDENTITY


def test_checker_protected_operators():
    """Protected operators are classified correctly."""
    checker = ConservationChecker()
    empty_topo = TopologyClassification(operator_name="test")

    for op_type, expected_basis in [
        (OperatorType.CONSERVED_CURRENT, ProtectionBasis.WARD_IDENTITY),
        (OperatorType.GAUGE_FIELD_STRENGTH, ProtectionBasis.BRST_SYMMETRY),
    ]:
        spec = OperatorSpec(
            name=f"test_{op_type.name}",
            op_type=op_type,
            lorentz_rank=2,
            spin_channel=2,
            external_momenta=2,
            mass_dimension=4,
            is_protected=True,
            protection_source=expected_basis.name,
        )
        report = checker.check(spec, empty_topo)
        assert report.verdict.is_protected is True, f"{op_type} should be protected, verdict={report.verdict}"


def test_checker_unprotected_operators():
    """Unprotected operators are classified correctly."""
    checker = ConservationChecker()
    empty_topo = TopologyClassification(operator_name="test")

    for op_type in [OperatorType.UNPROTECTED_FERMION, OperatorType.UNPROTECTED_SCALAR]:
        spec = OperatorSpec(
            name=f"test_{op_type.name}",
            op_type=op_type,
            lorentz_rank=0,
            spin_channel=0,
            external_momenta=2,
            mass_dimension=3,
            is_protected=False,
            protection_source="",
        )
        report = checker.check(spec, empty_topo)
        assert report.verdict.is_protected is False, f"{op_type} should be unprotected, verdict={report.verdict}"


def test_other_operator_type_not_protected():
    """OperatorType.OTHER should be unprotected."""
    checker = ConservationChecker()
    empty_topo = TopologyClassification(operator_name="test")
    spec = OperatorSpec(
        name="test_other",
        op_type=OperatorType.OTHER,
        lorentz_rank=0,
        spin_channel=0,
        external_momenta=0,
        mass_dimension=2,
        is_protected=False,
        protection_source="",
    )
    report = checker.check(spec, empty_topo)
    assert report.verdict.is_protected is False, f"OTHER should be unprotected, verdict={report.verdict}"


def run_tests():
    import traceback

    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_") and callable(o)]
    n_pass = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}")
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL: {name} - {e}")
        except Exception as e:
            print(f"  ERROR: {name} - {e}")
            traceback.print_exc()
    print(f"\n  {n_pass}/{len(tests)} passed")
    return n_pass == len(tests)


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_tests() else 1)
