"""
CGC Pipeline — Phase 2.1–2.6 Orchestrator
============================================================================

Runs the complete CGC classification pipeline for a given operator:
  1. Diagram Generation     (2.1)
  2. Momentum Classification (2.2)
  3. Topology Classification (2.3)
  4. Conservation-Law Check  (2.4)
  5. Resummation             (2.5)
  6. Benchmark Verification  (2.6)

The pipeline produces a CGCPipelineReport that contains all intermediate
results and the final verdict on injection-term nonzero-ness, ladder
resummation, and pole existence (within the ladder approximation).

Usage:
    from cgc import CGCPipeline
    from cgc.channels.tmunu_spin2 import TMunuSpin2

    operator = TMunuSpin2()
    pipeline = CGCPipeline()
    report = pipeline.run(operator)

    print(report.summary())
    report.export_json("output/tmunu_spin2_report.json")
"""


# References
#     CGC classification logic: Paper 1 (CG-Framework), Appendix E
#     Pipeline architecture: operator -> diagrams -> momentum -> topology -> conservation
#

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from .conservation_checker import ConservationChecker, ConservationReport
from .diagram_generator import (
    DiagramGenerator,
    DiagramSet,
    GeneratorBackend,
    OperatorSpec,
)
from .momentum_classifier import MomentumClassification, MomentumClassifier
from .resummation import ResummationResult, Resummator
from .topology_classifier import TopologyClassification, TopologyClassifier

# ── Pipeline Report ──────────────────────────────────────────────────────


@dataclass
class CGCPipelineReport:
    """Complete output of the CGC pipeline for one operator."""

    # Metadata
    operator: OperatorSpec
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    pipeline_version: str = field(
        default_factory=lambda: __import__("cgc").__version__
    )

    # Stage outputs
    diagram_set: DiagramSet | None = None
    momentum_classification: MomentumClassification | None = None
    topology_classification: TopologyClassification | None = None
    conservation_report: ConservationReport | None = None
    resummation_result: ResummationResult | None = None

    # Benchmark
    benchmark_passed: bool = False
    benchmark_details: dict = field(default_factory=dict)

    # Errors
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 72,
            f"CGC Pipeline Report: {self.operator.name}",
            f"  timestamp: {self.timestamp}",
            f"  version:   {self.pipeline_version}",
            "=" * 72,
            "",
        ]

        if self.errors:
            lines.append("ERRORS:")
            for e in self.errors:
                lines.append(f"  ❌ {e}")
            lines.append("")
            return "\n".join(lines)

        # Stage 1: Diagram Generation
        if self.diagram_set:
            lines.append("[1/5] Diagram Generation")
            lines.append(f"  backend:       {self.diagram_set.generator_backend.value}")
            lines.append(f"  total diagrams: {self.diagram_set.total_count}")
            lines.append(f"  max loops:     {self.diagram_set.max_loop_order}")
            lines.append(f"  complete:      {self.diagram_set.is_complete}")
            lines.append("")

        # Stage 2: Momentum Classification
        if self.momentum_classification:
            lines.append("[2/5] Momentum Transfer Classification")
            lines.append(self.momentum_classification.summary())
            lines.append("")

        # Stage 3: Topology Classification
        if self.topology_classification:
            lines.append("[3/5] Topology Classification")
            lines.append(self.topology_classification.summary())
            lines.append("")

        # Stage 4: Conservation Check
        if self.conservation_report:
            lines.append("[4/5] Conservation-Law Analysis")
            lines.append(self.conservation_report.summary())
            lines.append("")

        # Stage 5: Resummation
        if self.resummation_result:
            lines.append("[5/5] Resummation")
            lines.append(self.resummation_result.summary())
            lines.append("")

        # Benchmark
        lines.append(f"Benchmark: {'✅ PASSED' if self.benchmark_passed else '⚠ NOT VERIFIED'}")
        if self.benchmark_details:
            lines.append(f"  details: {json.dumps(self.benchmark_details, indent=2)}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize report to dict (for JSON export)."""
        d = {
            "operator_name": self.operator.name,
            "timestamp": self.timestamp,
            "pipeline_version": self.pipeline_version,
            "errors": self.errors,
            "benchmark_passed": self.benchmark_passed,
            "benchmark_details": self.benchmark_details,
        }

        if self.diagram_set:
            d["diagrams"] = {
                "total": self.diagram_set.total_count,
                "complete": self.diagram_set.is_complete,
                "backend": self.diagram_set.generator_backend.value,
            }

        if self.momentum_classification:
            d["momentum"] = {
                "q0_count": len(self.momentum_classification.zero_transfer),
                "q_nonzero_count": len(self.momentum_classification.nonzero_transfer),
                "suppression_ratio": self.momentum_classification.suppression_ratio,
            }

        if self.topology_classification:
            d["topology"] = {
                "single_bubble": len(self.topology_classification.single_bubble),
                "ladder": len(self.topology_classification.ladder),
                "crossed_ladder": len(self.topology_classification.crossed_ladder),
                "vertex_correction": len(self.topology_classification.vertex_correction),
            }

        if self.conservation_report:
            d["conservation"] = {
                "protected": self.conservation_report.verdict.is_protected,  # type: ignore[assignment]
                "basis": self.conservation_report.verdict.protection_basis.name,
                "matrix_element_nonzero": self.conservation_report.verdict.matrix_element_nonzero,
            }

        if self.resummation_result:
            d["resummation"] = {
                "injection_nonzero": self.resummation_result.injection_nonzero,  # type: ignore[assignment]
                "lambda_crit": self.resummation_result.lambda_crit,
                "is_critical": self.resummation_result.is_critical,
                "pole_exists": self.resummation_result.pole_exists,
            }

        return d

    def export_json(self, path: str) -> None:
        """Export report to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ── Pipeline ─────────────────────────────────────────────────────────────


class CGCPipeline:
    """
    Full CGC classification pipeline.

    Usage:
        pipeline = CGCPipeline(backend=GeneratorBackend.BUILTIN)
        operator = TMunuSpin2()  # or GaugeFieldStrength(), etc.
        report = pipeline.run(operator)
    """

    def __init__(self, backend: GeneratorBackend = GeneratorBackend.BUILTIN):
        self.backend = backend
        self.generator = DiagramGenerator(backend=backend)
        self.momentum_classifier = MomentumClassifier()
        self.topology_classifier = TopologyClassifier()
        self.conservation_checker = ConservationChecker()
        self.resummator = Resummator()

    def run(
        self,
        operator: OperatorSpec,
        max_loops: int = 1,
        Pi0_zero: float | None = None,
        lambda_eff: float | None = None,
        use_per_mode_dyson: bool = False,
    ) -> CGCPipelineReport:
        """
        Run the complete CGC pipeline.

        Args:
            operator: composite operator specification
            max_loops: maximum loop order (default 1)
            Pi0_zero: numeric Π₀(0) if available (for resummation)
            lambda_eff: numeric λ_eff if available (for resummation)

        Returns:
            CGCPipelineReport with all stage outputs
        """
        report = CGCPipelineReport(operator=operator)

        # ── Stage 1: Diagram Generation ──
        try:
            report.diagram_set = self.generator.generate(operator, max_loops)
        except Exception as e:
            report.errors.append(f"Stage 1 (diagram generation): {e}")
            return report

        # ── Stage 2: Momentum Classification ──
        try:
            report.momentum_classification = self.momentum_classifier.classify(report.diagram_set)
        except Exception as e:
            report.errors.append(f"Stage 2 (momentum classification): {e}")
            return report

        # ── Stage 3: Topology Classification ──
        try:
            report.topology_classification = self.topology_classifier.classify(
                report.momentum_classification,
                report.diagram_set.diagrams,
            )
        except Exception as e:
            report.errors.append(f"Stage 3 (topology classification): {e}")
            return report

        # ── Stage 4: Conservation Check ──
        try:
            report.conservation_report = self.conservation_checker.check(operator, report.topology_classification)
        except Exception as e:
            report.errors.append(f"Stage 4 (conservation check): {e}")
            return report

        # ── Stage 5: Resummation ──
        try:
            # Pi0(0): use the caller-provided value (flat-space single-bubble,
            # pi0_flat_continuum.py).  RP3 cross-validation is a SEPARATE
            # component (cgc/rp3_engine/) and is NOT invoked here.
            _Pi0 = Pi0_zero

            # Auto-crossed-ratio: compute r = Σ_crossed/(V·Π₀) from crossed_ladder_f2
            from .crossed_ladder_f2 import compute_crossed_ratio_explicit

            _crossed_ratio = None
            ot = operator.op_type.name
            if ot == "GAUGE_FIELD_STRENGTH":
                _crossed_ratio = compute_crossed_ratio_explicit(1.0, "F2")
            elif ot == "CONSERVED_CURRENT":
                # Tmunu: use degeneracy estimate from mode counting
                from .crossed_ladder_f2 import count_active_modes

                mc = count_active_modes(1.0)
                _crossed_ratio = 1.0 / mc.n_total if mc.n_total > 0 else None

            report.resummation_result = self.resummator.resummate(
                report.topology_classification,
                report.conservation_report,
                Pi0_zero=_Pi0,
                lambda_eff=lambda_eff,
                crossed_ratio=_crossed_ratio,
            )
        except Exception as e:
            report.errors.append(f"Stage 5 (resummation): {e}")
            return report

        return report

    def run_benchmark(self, operator: OperatorSpec, reference_data: dict) -> CGCPipelineReport:
        """
        Run pipeline AND verify against reference data (Phase 2.6).

        The benchmark verifies:
          1. Diagram count matches expected enumeration
          2. Momentum classification matches expected q=0/q≠0 split
          3. Topology classification matches expected ladder/bubble/other split
          4. Conservation verdict is correct
          5. Resummation produces correct critical condition form
        """
        report = self.run(operator)

        checks = {}

        # Check 1: Diagram count
        if "expected_total_diagrams" in reference_data:
            expected = reference_data["expected_total_diagrams"]
            actual = report.diagram_set.total_count if report.diagram_set else 0
            checks["diagram_count"] = actual == expected

        # Check 2: Momentum classification counts
        if "expected_q0_count" in reference_data:
            expected = reference_data["expected_q0_count"]
            actual = len(report.momentum_classification.zero_transfer) if report.momentum_classification else 0
            checks["q0_count"] = actual == expected

        if "expected_q_nonzero_count" in reference_data:
            expected = reference_data["expected_q_nonzero_count"]
            actual = len(report.momentum_classification.nonzero_transfer) if report.momentum_classification else 0
            checks["q_nonzero_count"] = actual == expected

        # Check 3: Topology classification counts
        if "expected_ladder_count" in reference_data:
            expected = reference_data["expected_ladder_count"]
            actual = len(report.topology_classification.ladder) if report.topology_classification else 0
            checks["ladder_count"] = actual == expected

        if "expected_bubble_count" in reference_data:
            expected = reference_data["expected_bubble_count"]
            actual = len(report.topology_classification.single_bubble) if report.topology_classification else 0
            checks["bubble_count"] = actual == expected

        # Check 4: Conservation verdict
        if "expected_protected" in reference_data:
            expected = reference_data["expected_protected"]
            actual = report.conservation_report.verdict.is_protected if report.conservation_report else None  # type: ignore[assignment]
            checks["conservation_protected"] = actual == expected

        # Check 5: Injection nonzero
        if "expected_injection_nonzero" in reference_data:  # type: ignore[assignment]
            expected = reference_data["expected_injection_nonzero"]  # type: ignore[assignment]
            actual = report.resummation_result.injection_nonzero if report.resummation_result else None  # type: ignore[assignment]
            checks["injection_nonzero"] = actual == expected  # type: ignore[assignment]

        report.benchmark_details = checks
        report.benchmark_passed = all(checks.values()) if checks else False

        return report
