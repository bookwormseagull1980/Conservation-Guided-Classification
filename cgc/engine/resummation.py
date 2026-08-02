"""
Resummation Module — Phase 2.5
============================================================================

Performs geometric series resummation of ladder diagrams and computes
the critical condition for spectral pole formation.

Core logic:
  Ladder diagrams with q=0 and protected matrix elements form a geometric
  series in the number of bubbles:
    Π(q=0) = Π₀ + Π₀·(V·Π₀) + Π₀·(V·Π₀)² + …
           = Π₀ / (1 − V·Π₀)

  The spectral function develops a δ-pole when the denominator vanishes:
    1 − V·Π₀(0) = 0   →   critical condition

  More precisely, the injection term Δ_σ receives contributions from
  each rung of the ladder. The resummation encodes the accumulation of
  spectral weight at μ²=0 as σ → σ_c.

Output:
  - Critical coupling λ_crit
  - Pole residue (if pole forms)
  - Continuum correction from single-bubble diagrams

Reference: Appendix E, Eqs. (E.3)–(E.5).
"""


# References
#     Ladder resummation: Dyson-Schwinger equation formalism
#     Roberts-Williams (1994), Prog. Part. Nucl. Phys. 33, 477
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .conservation_checker import ConservationReport
from .topology_classifier import TopologyClassification

# ── Resummation Result ───────────────────────────────────────────────────


@dataclass
class ResummationResult:
    """Output of geometric series resummation of ladder diagrams."""

    # Inputs
    operator_name: str
    n_ladder_diagrams: int

    # Single-bubble contribution at q=0
    Pi0_at_zero: float | None = None  # Π₀(μ²=0)
    Pi0_symbolic: str = ""  # symbolic expression

    # Effective coupling of the irreducible insertion
    lambda_eff: float | None = None  # V ≈ λ_eff
    lambda_eff_symbolic: str = ""

    # Geometric series
    denominator: float | None = None  # 1 − V·Π₀(0)
    geometric_factor: float | None = None  # 1/(1 − V·Π₀(0))

    # Critical condition
    lambda_crit: float | None = None  # λ_crit = 1/Π₀(0)
    is_critical: bool = False  # |λ_eff| ≥ |λ_crit| ?
    distance_to_critical: float | None = None  # (λ_eff − λ_crit)/λ_crit

    # Pole properties (if critical)
    pole_residue: float | None = None  # Z = residue of the pole
    pole_exists: bool = False

    # Crossed ladder
    crossed_ratio: float | None = None  # r = Σ_crossed / (V·Π₀)

    # Injection term
    injection_nonzero: bool = False  # guaranteed by conservation?

    # Notes
    warnings: list[str] | None = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []

    def summary(self) -> str:
        lines = [
            f"Resummation Analysis: {self.operator_name}",
            f"  ladder diagrams:     {self.n_ladder_diagrams}",
            f"  crossed ratio:       {self._fmt(self.crossed_ratio)}",
            f"  injection nonzero:   {self.injection_nonzero}",
            f"  Π₀(0):               {self._fmt(self.Pi0_at_zero)} {self.Pi0_symbolic}",
            f"  λ_eff:               {self._fmt(self.lambda_eff)} {self.lambda_eff_symbolic}",
            f"  1 − V·Π₀(0):         {self._fmt(self.denominator)}",
            f"  λ_crit:              {self._fmt(self.lambda_crit)}",
            f"  is critical:         {self.is_critical}",
            f"  Δλ/λ_crit:           {self._fmt(self.distance_to_critical)}",
            f"  pole exists:         {self.pole_exists}",
        ]
        if self.warnings:
            lines.append("  warnings:")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")
        return "\n".join(lines)

    @staticmethod
    def _fmt(val: Any) -> str:
        if val is None:
            return "—"
        return f"{val:.6g}"


# ── Resummation Engine ───────────────────────────────────────────────────


class Resummator:
    """
    Performs geometric series resummation of ladder diagrams.

    The resummation is EXACT within the ladder approximation:
      Δ_σ(μ²) = Π₀(μ²) / [1 − V·Π₀(μ²)]

    The critical condition 1 − V·Π₀(0) = 0 is a necessary condition
    for δ-pole formation. Whether the flow actually reaches this
    condition is a dynamical question answered by the FRG flow, not
    by this module.

    Two modes:
      - symbolic:  output symbolic expressions for Π₀, V, and the series
      - numerical: compute Π₀(0) and λ_crit from input values
    """

    def resummate(
        self,
        topo: TopologyClassification,
        cons: ConservationReport,
        Pi0_zero: float | None = None,
        lambda_eff: float | None = None,
        crossed_ratio: float | None = None,
    ) -> ResummationResult:
        """
        Compute the geometric series resummation and critical condition.

        When crossed_ratio is provided, the denominator includes crossed
        ladder contributions:
            1 − V·Π₀ − Σ_crossed
          = 1 − V·Π₀ − r_crossed · V·Π₀
          = 1 − (1 + r_crossed) · V·Π₀

        This reduces the effective critical coupling:
            λ_crit = 1 / [(1 + r_crossed) · Π₀]

        Args:
            topo: topology classification
            cons: conservation report
            Pi0_zero: numeric value of Π₀(μ²=0) if available
            lambda_eff: numeric value of V if available
            crossed_ratio: r = Σ_crossed / (V·Π₀) — ratio of crossed-ladder
                           to ladder contribution. None = ladder-only.

        Returns:
            ResummationResult with critical condition and pole status
        """
        n_ladder = len(topo.ladder)

        result = ResummationResult(
            operator_name=cons.operator.name,
            n_ladder_diagrams=n_ladder,
            injection_nonzero=cons.verdict.matrix_element_nonzero,
            crossed_ratio=crossed_ratio,
        )

        # ── Single-bubble contribution ──
        if Pi0_zero is not None:
            result.Pi0_at_zero = Pi0_zero
        result.Pi0_symbolic = self._pi0_symbolic(cons)

        # ── Effective coupling ──
        if lambda_eff is not None:
            result.lambda_eff = lambda_eff
        result.lambda_eff_symbolic = self._lambda_symbolic(cons)

        # ── Effective enhancement factor from crossed ladder ──
        enhancement_factor = 1.0
        if crossed_ratio is not None:
            enhancement_factor = 1.0 + crossed_ratio

        # ── Geometric series ──
        if Pi0_zero is not None and lambda_eff is not None:
            result.denominator = 1.0 - enhancement_factor * lambda_eff * Pi0_zero
            if abs(result.denominator) > 1e-15:
                result.geometric_factor = 1.0 / result.denominator
            else:
                result.warnings.append(  # type: ignore[union-attr]
"Denominator 1 − (1+r)·V·Π₀(0) ≈ 0 → pole condition met!")
                result.geometric_factor = float("inf")
        else:
            result.denominator = None
            result.geometric_factor = None
            result.warnings.append(  # type: ignore[union-attr]

                "Numerical Π₀(0) and λ_eff not provided — critical condition cannot be evaluated numerically."
            )

        # ── Critical coupling ──
        # With crossed ladder: λ_crit = 1 / [(1+r) · Π₀]
        if Pi0_zero is not None and Pi0_zero != 0:
            result.lambda_crit = 1.0 / (enhancement_factor * Pi0_zero)
            if lambda_eff is not None:
                result.distance_to_critical = (lambda_eff - result.lambda_crit) / result.lambda_crit
                result.is_critical = abs(lambda_eff) >= abs(result.lambda_crit)

        # ── Pole existence ──
        # A pole forms if:
        #   1. Injection is nonzero (conservation guarantee)  ✓/✗
        #   2. Ladder diagrams exist                          ✓/✗
        #   3. Critical condition is reached                  ? (FRG question)
        # Pole exists if: (1) injection nonzero (conservation guarantee)
        # AND (2) critical condition reached.
        # Note: n_ladder counts explicit ladder diagrams in the input set
        # (non-zero only at L >= 2). The resummation itself is the
        # geometric series that generates the ladder — the pole condition
        # depends on the denominator vanishing, not on explicit ladder count.
        if result.injection_nonzero and result.is_critical:
            result.pole_exists = True
            # Residue: Z = lim_{μ²→0} μ²·Δ_σ(μ²)
            # In the ladder approximation: Z = Π₀(0)/[V·Π₀'(0)]
            # (requires derivative Π₀'(0) — not computed here)
            result.pole_residue = None  # requires Π₀'(0)
            result.warnings.append(  # type: ignore[union-attr]
"Pole residue requires Π₀'(μ²=0) — not computed.")

        return result

    # ── Symbolic Helpers ──

    def _pi0_symbolic(self, cons: ConservationReport) -> str:
        """Symbolic form of the single-bubble contribution at q=0."""
        op = cons.operator
        if op.op_type.name in ("CONSERVED_CURRENT",):
            # For Tμν: Π₀(0) ∝ a₂ (Seeley-DeWitt) × (spin-2 projection)
            # The exact coefficient depends on the field content
            return "Π₀(0) = c_eff · a₂ / (16π²)  [see Appendix D]"
        if op.op_type.name == "GAUGE_FIELD_STRENGTH":
            return "Π₀(0) = N_c · g² / (48π²)  [Yang-Mills, one-loop]"
        if op.op_type.name in ("UNPROTECTED_SCALAR", "UNPROTECTED_FERMION"):
            return "Π₀(0) → 0 (unprotected — suppressed by powers of μ²/Λ²)"
        return "Π₀(0) — unknown (operator type not in database)"

    def _lambda_symbolic(self, cons: ConservationReport) -> str:
        """Symbolic form of the effective coupling V."""
        op = cons.operator
        if op.op_type.name in ("CONSERVED_CURRENT",):
            return "λ_eff = V(q=0) — determined by FRG trace density"
        if op.op_type.name == "GAUGE_FIELD_STRENGTH":
            return "λ_eff ∝ g² — determined by gauge coupling RG flow"
        return "λ_eff — unknown"

    # ── Error Estimation ──

    def estimate_ladder_accuracy(
        self,
        result: ResummationResult,
        non_ladder_contrib: float | None = None,
    ) -> dict[str, float]:
        """
        Estimate the accuracy of the ladder approximation.

        If non-ladder diagram contributions (crossed ladder, vertex
        corrections) are available, compute their relative size compared
        to the ladder series.

        Args:
            result: resummation result
            non_ladder_contrib: estimated contribution from non-ladder diagrams

        Returns:
            dict with {relative_error, dominance_ratio, ...}
        """
        estimates: dict[str, float] = {}

        if non_ladder_contrib is not None and result.geometric_factor is not None and result.geometric_factor != 0 and result.geometric_factor != float("inf"):
                relative_error = abs(non_ladder_contrib / result.geometric_factor)
                estimates["relative_error"] = relative_error
                estimates["dominance_ratio"] = (
                    abs(result.geometric_factor / non_ladder_contrib) if non_ladder_contrib != 0 else float("inf")
                )

        return estimates
