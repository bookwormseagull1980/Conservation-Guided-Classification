"""Phase 2 Tests: Gauge Field Strength, Fermion Bilinear, Higgs Quartic.

Covers the three Phase 2 channels beyond the Tμν benchmark.
Also tests the resummation engine for protected vs unprotected operators.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgc.engine.diagram_generator import DiagramGenerator, OperatorType
from cgc.engine.momentum_classifier import MomentumClassifier
from cgc.engine.conservation_checker import ConservationChecker
from cgc.engine.topology_classifier import TopologyClassifier
from cgc.engine.resummation import Resummator
from cgc.channels.gauge_field import GaugeFieldStrength, GAUGE_FIELD_BENCHMARK
from cgc.channels.fermion_bilinears import FermionBilinears, FERMION_BENCHMARK
from cgc.channels.higgs_quartic import HiggsQuartic, HIGGS_QUARTIC_BENCHMARK


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gauge Field Strength Channel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGaugeFieldStrength:
    """Fμν^a F^{aμν} — Phase 2 primary target channel."""

    @pytest.fixture
    def op(self):
        return GaugeFieldStrength("SU(3)")

    def test_diagram_count(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        assert ds.total_count == 4, f"Expected 4, got {ds.total_count}"

    def test_q0_count(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        assert len(cl.zero_transfer) == 2

    def test_q_nonzero_count(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        assert len(cl.nonzero_transfer) == 2

    def test_bubble_count(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        assert len(topo.single_bubble) == 2

    def test_no_ladder_one_loop(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        assert len(topo.ladder) == 0, "No ladders at one-loop"

    def test_brst_protected(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        cc = ConservationChecker()
        report = cc.check(op, topo)
        assert report.verdict.is_protected, "Gauge field strength is BRST-protected"
        assert report.verdict.protection_basis.name == "BRST_SYMMETRY"

    def test_injection_nonzero(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        cc = ConservationChecker()
        report = cc.check(op, topo)
        resum = Resummator()
        result = resum.resummate(topo, report)
        assert result.injection_nonzero, "Protected → injection nonzero"

    def test_benchmark_match(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        cc = ConservationChecker()
        report = cc.check(op, topo)
        resum = Resummator()
        result = resum.resummate(topo, report)

        assert ds.total_count == GAUGE_FIELD_BENCHMARK["expected_total_diagrams"]
        assert len(cl.zero_transfer) == GAUGE_FIELD_BENCHMARK["expected_q0_count"]
        assert len(cl.nonzero_transfer) == GAUGE_FIELD_BENCHMARK["expected_q_nonzero_count"]
        assert len(topo.single_bubble) == GAUGE_FIELD_BENCHMARK["expected_bubble_count"]
        assert len(topo.ladder) == GAUGE_FIELD_BENCHMARK["expected_ladder_count"]
        assert report.verdict.is_protected == GAUGE_FIELD_BENCHMARK["expected_protected"]
        assert result.injection_nonzero == GAUGE_FIELD_BENCHMARK["expected_injection_nonzero"]

    def test_field_types(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        field_labels = set()
        for d in ds.diagrams:
            for v in d.vertices:
                for f in v.fields:
                    label = f.rsplit("_", 1)[0].replace("fast_", "")
                    field_labels.add(label)
        assert "gauge_boson" in field_labels, "Gauge self-coupling bubble required"
        assert "fermion" in field_labels or "weyl_fermion" in field_labels, "Quark bubble required"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fermion Bilinear Channel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFermionBilinear:
    """ψ̄Γψ — Phase 2 unprotected operator (null test)."""

    @pytest.fixture
    def op(self):
        return FermionBilinears("scalar")

    def test_diagram_count(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        assert ds.total_count == 2

    def test_q0_count(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        assert len(cl.zero_transfer) == 1

    def test_unprotected(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        cc = ConservationChecker()
        report = cc.check(op, topo)
        assert not report.verdict.is_protected

    def test_injection_false(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        cc = ConservationChecker()
        report = cc.check(op, topo)
        resum = Resummator()
        result = resum.resummate(topo, report)
        assert not result.injection_nonzero, "Unprotected → injection suppressed"

    def test_no_pole(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        cc = ConservationChecker()
        report = cc.check(op, topo)
        resum = Resummator()
        result = resum.resummate(topo, report)
        assert not result.pole_exists

    def test_benchmark_match(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        cc = ConservationChecker()
        report = cc.check(op, topo)
        resum = Resummator()
        result = resum.resummate(topo, report)

        assert ds.total_count == FERMION_BENCHMARK["expected_total_diagrams"]
        assert len(cl.zero_transfer) == FERMION_BENCHMARK["expected_q0_count"]
        assert report.verdict.is_protected == FERMION_BENCHMARK["expected_protected"]
        assert result.injection_nonzero == FERMION_BENCHMARK["expected_injection_nonzero"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Higgs Quartic Channel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHiggsQuartic:
    """λ(φ†φ)² — Phase 2 unprotected operator."""

    @pytest.fixture
    def op(self):
        return HiggsQuartic()

    def test_diagram_count(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        assert ds.total_count == 4

    def test_q0_count(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        assert len(cl.zero_transfer) == 2

    def test_unprotected(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        cc = ConservationChecker()
        report = cc.check(op, topo)
        assert not report.verdict.is_protected

    def test_injection_false(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        cc = ConservationChecker()
        report = cc.check(op, topo)
        resum = Resummator()
        result = resum.resummate(topo, report)
        assert not result.injection_nonzero, "Unprotected → injection suppressed"

    def test_no_pole(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        cc = ConservationChecker()
        report = cc.check(op, topo)
        resum = Resummator()
        result = resum.resummate(topo, report)
        assert not result.pole_exists, "Unprotected → no spectral pole"

    def test_benchmark_match(self, op):
        gen = DiagramGenerator()
        ds = gen.generate(op, max_loops=1)
        mc = MomentumClassifier()
        cl = mc.classify(ds)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        tc = TopologyClassifier()
        topo = tc.classify(cl, ds.diagrams)
        cc = ConservationChecker()
        report = cc.check(op, topo)
        resum = Resummator()
        result = resum.resummate(topo, report)

        assert ds.total_count == HIGGS_QUARTIC_BENCHMARK["expected_total_diagrams"]
        assert len(cl.zero_transfer) == HIGGS_QUARTIC_BENCHMARK["expected_q0_count"]
        assert report.verdict.is_protected == HIGGS_QUARTIC_BENCHMARK["expected_protected"]
        assert result.injection_nonzero == HIGGS_QUARTIC_BENCHMARK["expected_injection_nonzero"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cross-Channel Physics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.mark.physics
class TestPhase2Physics:
    """Physics-level tests spanning all Phase 2 channels."""

    def test_protected_have_nonzero_injection(self):
        """Key CGC prediction: protected operators → nonzero injection."""
        for op_factory in [GaugeFieldStrength, lambda: GaugeFieldStrength("SU(3)")]:
            op = op_factory()
            gen = DiagramGenerator()
            ds = gen.generate(op, max_loops=1)
            mc = MomentumClassifier()
            cl = mc.classify(ds)
            tc = TopologyClassifier()
            topo = tc.classify(cl, ds.diagrams)
            cc = ConservationChecker()
            report = cc.check(op, topo)
            resum = Resummator()
            result = resum.resummate(topo, report)
            assert result.injection_nonzero == report.verdict.is_protected, \
                "Conservation protection ⇔ injection nonzero"

    def test_unprotected_have_zero_injection(self):
        """Key CGC prediction: unprotected operators → suppressed injection."""
        for op_factory in [FermionBilinears, lambda: FermionBilinears("scalar"),
                          HiggsQuartic]:
            op = op_factory()
            gen = DiagramGenerator()
            ds = gen.generate(op, max_loops=1)
            mc = MomentumClassifier()
            cl = mc.classify(ds)
            tc = TopologyClassifier()
            topo = tc.classify(cl, ds.diagrams)
            cc = ConservationChecker()
            report = cc.check(op, topo)
            resum = Resummator()
            result = resum.resummate(topo, report)
            assert not result.injection_nonzero, \
                f"{op.name}: unprotected → injection suppressed"

    def test_suppression_ratio_universal(self):
        """All channels at one-loop: exactly 50% of diagrams are q=0."""
        for op_factory in [GaugeFieldStrength, FermionBilinears,
                          HiggsQuartic, lambda: GaugeFieldStrength("SU(3)"),
                          lambda: FermionBilinears("scalar")]:
            op = op_factory()
            gen = DiagramGenerator()
            ds = gen.generate(op, max_loops=1)
            mc = MomentumClassifier()
            cl = mc.classify(ds)
            ratio = len(cl.zero_transfer) / ds.total_count
            assert ratio == 0.5, f"{op.name}: suppression ratio {ratio:.2%} ≠ 50%"

    def test_cgc_protection_dichotomy(self):
        """The CGC method predicts a sharp dichotomy:
        Protected → potential pole; Unprotected → never a pole."""
        from cgc.channels.tmunu_spin2 import TMunuSpin2

        protected_channels = [TMunuSpin2, GaugeFieldStrength,
                             lambda: GaugeFieldStrength("SU(3)")]
        unprotected_channels = [FermionBilinears, HiggsQuartic,
                               lambda: FermionBilinears("scalar")]

        for op_factory in protected_channels:
            op = op_factory()
            gen = DiagramGenerator()
            ds = gen.generate(op, max_loops=1)
            mc = MomentumClassifier()
            cl = mc.classify(ds)
            tc = TopologyClassifier()
            topo = tc.classify(cl, ds.diagrams)
            cc = ConservationChecker()
            report = cc.check(op, topo)
            assert report.verdict.is_protected, \
                f"{op.name}: should be protected"

        for op_factory in unprotected_channels:
            op = op_factory()
            gen = DiagramGenerator()
            ds = gen.generate(op, max_loops=1)
            mc = MomentumClassifier()
            cl = mc.classify(ds)
            tc = TopologyClassifier()
            topo = tc.classify(cl, ds.diagrams)
            cc = ConservationChecker()
            report = cc.check(op, topo)
            assert not report.verdict.is_protected, \
                f"{op.name}: should NOT be protected"
