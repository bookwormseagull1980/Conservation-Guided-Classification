"""FRG Flow of Effective 4-Operator Coupling on RP³.

CGC Phase 3: solves the Wetterich flow equation for the dimensionless
effective coupling V_k of the composite operator (F² or Tμν).

Core question:
  Does the RG flow drive V from the perturbative UV value
  (~g²/(16π²) ∼ 10⁻³) to the critical value λ_crit (∼10–30)
  required for spectral pole formation?

Physical setup
--------------
The ladder resummation predicts a pole in the two-point function when:
    1 − V_k · Π₀(k²=0) → 0
    → V_crit = 1/Π₀(0) = λ_crit

The one-loop Π₀(0) is computed by FRGTraceDensity:
  - F²:  Π₀(0) ≈ 3.57×10⁻² → λ_crit ≈ 28
  - Tμν: Π₀(0) ≈ 9.96×10⁻² → λ_crit ≈ 10

The bare UV coupling:
  V_UV ∼ g²/(16π²) ∼ 0.246/(16π²) ≈ 1.56×10⁻³  (for F²)
  V_UV ∼ (1/L⁴)/(16π²) ∼ 2.8×10⁻³               (for Tμν on RP³)

Gap to critical: factor ∼ 3×10³–2×10⁴.

Three mechanisms that can enhance V in the IR:
  1. Anomalous dimension: η_V < 0 makes V relevant
  2. RP³ curvature: discrete Laplacian spectrum modifies threshold integrals
  3. Non-perturbative feedback: ∂_t V ∝ V² → rapid growth near critical

Method
------
Wetterich equation in LPA:
  ∂_t V_k = −½ ∂̃_t Tr[ (Γ_k^(2) + R_k)^(−1) · V_k^(4) ]

Projected onto the 4-operator vertex:
  ∂_t V_k = V_k² · I(k) / (16π²)
         + V_k³ · J(k) / (16π²)²  (next order)
         + η_V(k) · V_k            (anomalous dimension, if present)

where I(k) and J(k) are dimensionless threshold integrals determined
by the trace density with the Litim optimal regulator.

On RP³, the momentum integral becomes a discrete sum over eigenvalues
λ_n = n(n+2)/L², which can enhance I(k) at scales k ≲ M_P/L.

Author: CGC Phase 3
Date: 2026-07-29
"""


# References
#     Beta function integration: Wetterich (1993)
#     LPA approximation: Berges-Tetradis-Wetterich (2002)
#

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frg_trace_density import (
    G3_MG,
    M_P,
    FRGTraceDensity,
)
from cgc.params import L_RP3

# ═══════════════════════════════════════════════════════════════
# Physical scales
# ═══════════════════════════════════════════════════════════════

K_UV = M_P  # UV cutoff = Planck scale
K_IR = 1.0  # IR cutoff = 1 GeV
M_CURV = M_P / L_RP3  # Curvature effective mass scale [GeV] ≈ M_G  (L from cgc/params.py)


@dataclass
class FlowConfig:
    """Configuration for the FRG flow solver."""

    operator_name: str = "F²"
    k_uv: float = K_UV  # UV cutoff [GeV]
    k_ir: float = K_IR  # IR cutoff [GeV]
    n_grid: int = 500  # log-spaced grid points
    v_uv: float = 1.56e-3  # bare V at UV (perturbative)
    lambda_crit: float = 28.0  # critical coupling for pole
    include_rp3: bool = True  # RP³ curvature effects
    include_anomalous_dim: bool = True  # η_V from trace density
    include_v3: bool = True  # V³ term (next-to-leading)
    verbose: bool = False


@dataclass
class FlowResult:
    """Result of FRG flow integration."""

    operator_name: str
    k_grid: np.ndarray  # momentum scales [GeV]
    v_grid: np.ndarray  # V(k) along the flow
    eta_grid: np.ndarray  # anomalous dimension η_V(k)
    beta_grid: np.ndarray  # beta function ∂_t V(k)
    v_ir: float  # V at IR cutoff
    v_uv: float  # V at UV cutoff (for verification)
    crosses_critical: bool  # does V cross λ_crit?
    k_cross: float | None  # scale where V = λ_crit (if any)
    log_enhancement: float  # ln(V_IR / V_UV)
    contributions: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class FRGFlowSolver:
    """Solve the FRG flow equation for the effective 4-operator coupling.

    Flow equation:
      ∂_t V_k = Σ_i [contribution_i(V_k, k)]

    Contributions:
      1. Canonical: 0 (V is dimensionless in d=4)
      2. V² term: from trace density with Litim regulator
           ∂_t V^(2) = V² · I(k)
           I(k) = ∂_t[Π₀]_k / 2
         where Π₀(k) is the integrated trace density up to scale k.
      3. V³ term: vertex dressing at two-loop
           ∂_t V^(3) = V³ · J(k) / (16π²)
      4. Anomalous dimension: if present from curvature feedback
           ∂_t V^(η) = η_V(k) · V

    On RP³, the integration measure is modified by the discrete
    Laplacian spectrum below the curvature scale.
    """

    def __init__(self, config: FlowConfig):
        self.cfg = config
        self._k_grid = np.geomspace(config.k_ir, config.k_uv, config.n_grid)
        self._frg = FRGTraceDensity(k_ir=config.k_ir, k_uv=config.k_uv, n_grid=config.n_grid)

    # ═══════════════════════════════════════════════════════════
    # Threshold integral I(k) — from trace density
    # ═══════════════════════════════════════════════════════════

    def compute_threshold_integral(self, k: float) -> float:
        r"""I(k): dimensionless V² coefficient at scale k.

        I(k) = ∂(Π₀)_k / (∂ ln k) × (scale factor)

        The trace density η(k) = dΠ₀/d(ln k) gives the derivative.
        At scale k:
          I(k) = η(k) / 2  [the factor 1/2 is from the V² β-function]

        This is the ONE-LOOP contribution. On RP³, the discrete
        spectrum enhances η(k) for k ≲ M_CURV.
        """
        # Use the Litim threshold function to compute η(k) directly
        # η(k) = Σ_f d_f · vertex² · T(k/m_f) / (16π²)
        # at the SCALE k (not integrated)

        vertex_sq = self._get_vertex_sq()

        # Sum over all fields
        total_eta = 0.0
        fields = self._get_fields()

        for f_entry in fields:
            dof = f_entry["dof"]
            mass = f_entry["mass_gev"]
            # Litim threshold: 2k²/(k²+m²) — unified with frg_flow_rp3.py (2026-07-30)
            T_k = 2.0 * k * k / (k * k + mass**2)
            total_eta += dof * vertex_sq * T_k / (16.0 * np.pi**2)

        # NOTE: rp3_factor removed (2026-07-30).
        # The manual curvature enhancement 1+3/(1+x²) had no first-principles derivation.
        # For RP3 discrete spectrum effects, use frg_flow_rp3.py which performs
        # proper mode summation over the Camporesi spectrum.

        return total_eta / 2.0  # I(k) = η(k)/2

    def _get_vertex_sq(self) -> float:
        """Effective vertex squared at the UV."""
        if "F2" in self.cfg.operator_name or "F²" in self.cfg.operator_name:
            return G3_MG**2  # g₃²
        return 1.0 / L_RP3**4  # 1/L⁴ for Tμν on RP³

    def _get_fields(self) -> list[dict]:
        """Get field list for this operator."""
        if "F2" in self.cfg.operator_name or "F²" in self.cfg.operator_name:
            frg = self._frg
            return [{"name": f.name, "dof": f.dof, "mass_gev": f.mass_gev} for f in frg.fields_f2()]
        frg = self._frg
        return [{"name": f.name, "dof": f.dof, "mass_gev": f.mass_gev} for f in frg.fields_tmunu()]

    # ═══════════════════════════════════════════════════════════
    # Anomalous dimension η_V(k)
    # ═══════════════════════════════════════════════════════════

    def compute_anomalous_dimension(self, k: float, V: float) -> float:
        r"""Anomalous dimension η_V(k) from feedback.

        The anomalous dimension arises from the V-dependence of the
        propagator in the Wetterich equation:

          η_V(k) = −d ln Z_V / d ln k

        where Z_V is the wavefunction renormalization of the operator.
        In the self-consistent approximation:

          η_V(k) ≈ −c · V · I(k)

        where c ∼ O(1) is a diagrammatic coefficient.
        The negative sign means growing V → negative η_V → V becomes
        MORE relevant → faster growth (positive feedback).

        At the exact FRG level:
          η_V = (2/d) · V · ∂_k [∂_p² Π_k(p)]_{p=0}

        We approximate this with the Litim-derived threshold.
        """
        if not self.cfg.include_anomalous_dim:
            return 0.0

        # Self-consistent: η_V ∝ V · I(k)
        # The coefficient comes from the derivative of the loop integral
        # w.r.t. external momentum. For a scalar-like bubble:
        #   η_V ≈ −(1/6) · V · I(k) · (k²/⟨m²⟩_eff)
        thr_I = self.compute_threshold_integral(k)
        # Effective mass scale at this k
        m_eff_sq = max(k**2, M_CURV**2) if self.cfg.include_rp3 else k**2

        # Coefficient: from p-derivative of the loop integral
        # At k >> m: suppression ~ k^2/m^2, but I(k) already small
        # At k ~ m: O(1) coupling
        suppression = k**2 / (k**2 + m_eff_sq)
        return -0.5 * V * thr_I * suppression

    # ═══════════════════════════════════════════════════════════
    # V³ coefficient J(k) — two-loop vertex correction
    # ═══════════════════════════════════════════════════════════

    def compute_v3_coefficient(self, k: float) -> float:
        r"""J(k): dimensionless V³ coefficient at scale k.

        The V³ term comes from two-loop vertex corrections.
        J(k) = (phase space integral for 2-loop dressing) / (16π²)

        Approximate by nested one-loop integrals:
          J(k) ≈ [I(k)]² × (phase space overlap factor)

        The overlap factor accounts for the fact that the two loops
        share momentum — typically O(1/2).
        """
        thr_I = self.compute_threshold_integral(k)
        overlap = 0.5  # typical phase-space overlap for nested loops
        return thr_I * thr_I * overlap / (16.0 * np.pi**2)

    # ═══════════════════════════════════════════════════════════
    # Beta function
    # ═══════════════════════════════════════════════════════════

    def beta(self, k: float, V: float) -> float:
        r"""Beta function: ∂_t V = dV/d(ln k).

        ∂_t V = V² · I(k)                    [one-loop V²]
               + V³ · J(k)                   [two-loop V³]
               + η_V(k) · V                  [anomalous dimension]
               + curvature_enhancement(V, k) [RP³ discrete spectrum]
        """
        beta_val = 0.0

        # V² term (always present)
        thr_I = self.compute_threshold_integral(k)
        beta_val += V**2 * thr_I

        # V³ term (next-to-leading)
        if self.cfg.include_v3:
            J = self.compute_v3_coefficient(k)
            beta_val += V**3 * J

        # Anomalous dimension
        eta = self.compute_anomalous_dimension(k, V)
        beta_val += eta * V

        return beta_val

    # ═══════════════════════════════════════════════════════════
    # Integration
    # ═══════════════════════════════════════════════════════════

    def solve(self) -> FlowResult:
        """Integrate the flow equation from UV to IR.

        Uses a 4th-order Runge-Kutta integrator along the
        log-spaced grid from k_UV down to k_IR.

        The flow direction is from UV → IR (dt = d(ln k) < 0).
        We integrate backwards: V(k_{i+1}) → V(k_i) for k_i < k_{i+1}.
        """
        n = self.cfg.n_grid
        k_grid = self._k_grid
        V = np.zeros(n)
        eta_grid = np.zeros(n)
        beta_grid = np.zeros(n)

        # Initial condition at UV
        V[0] = self.cfg.v_uv
        eta_grid[0] = 0.0
        beta_grid[0] = 0.0

        d_ln_k = np.log(k_grid[1] / k_grid[0])

        # Integrate from UV (index 0, large k) to IR (index n-1, small k)
        # Flow: dt = d(ln k), we step from large ln k to small ln k
        # dV/d(ln k) = +beta  [positive for growth toward IR]
        for i in range(1, n):
            k_current = k_grid[i - 1]  # larger k (UV side)
            k_next = k_grid[i]  # smaller k (IR side)
            V_current = V[i - 1]

            # RK4 step: dt = ln(k_next) - ln(k_current) < 0
            # But beta is dV/d(ln k), so for decreasing ln k:
            # V_next = V_current + beta * (ln(k_next) - ln(k_current))
            # where (ln(k_next) - ln(k_current)) = +d_ln_k if we step IRward

            # Actually, we're stepping from UV to IR, so k decreases
            # d(ln k) < 0, but we define the step direction explicitly:
            dt = -d_ln_k  # negative: UV → IR

            # RK4
            k_mid = np.sqrt(k_current * k_next)

            k1 = self.beta(k_current, V_current)
            k2 = self.beta(k_mid, V_current + 0.5 * dt * k1)
            k3 = self.beta(k_mid, V_current + 0.5 * dt * k2)
            k4 = self.beta(k_next, V_current + dt * k3)

            V[i] = V_current + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Stability: clip to physical range
            V[i] = max(V[i], 1e-30)

            eta_grid[i] = self.compute_anomalous_dimension(k_next, V[i])
            beta_grid[i] = self.beta(k_next, V[i])

        # Check for crossing
        crosses = np.any(self.cfg.lambda_crit <= V)
        k_cross = None
        if crosses:
            cross_idx = np.argmax(self.cfg.lambda_crit <= V)
            k_cross = k_grid[cross_idx]

        result = FlowResult(
            operator_name=self.cfg.operator_name,
            k_grid=k_grid,
            v_grid=V,
            eta_grid=eta_grid,
            beta_grid=beta_grid,
            v_ir=V[-1],
            v_uv=V[0],
            crosses_critical=bool(crosses),
            k_cross=k_cross,
            log_enhancement=np.log(V[-1] / V[0]),
        )

        # Contributions breakdown
        I_ir = self.compute_threshold_integral(k_grid[-1])
        eta_ir = self.compute_anomalous_dimension(k_grid[-1], V[-1])
        J_ir = self.compute_v3_coefficient(k_grid[-1])
        beta_v2 = V[-1] ** 2 * I_ir
        beta_v3 = V[-1] ** 3 * J_ir
        beta_eta = eta_ir * V[-1]

        result.contributions = [
            {"name": "V² term", "value": beta_v2, "fraction": beta_v2 / (beta_grid[-1] + 1e-60)},
            {"name": "V³ term", "value": beta_v3, "fraction": beta_v3 / (beta_grid[-1] + 1e-60)},
            {"name": "η_V term", "value": beta_eta, "fraction": beta_eta / (beta_grid[-1] + 1e-60)},
        ]

        # Notes
        result.notes.append(f"UV initial value: V(k_UV={self.cfg.k_uv:.2e} GeV) = {V[0]:.6e}")
        result.notes.append(f"IR final value:   V(k_IR={self.cfg.k_ir:.2e} GeV) = {V[-1]:.6e}")
        result.notes.append(f"Log enhancement: ln(V_IR/V_UV) = {result.log_enhancement:.4f}")
        result.notes.append(f"λ_crit = {self.cfg.lambda_crit:.2f}")
        if crosses:
            result.notes.append(f"✓ CROSSES CRITICAL at k = {k_cross:.4e} GeV")
        else:
            gap = self.cfg.lambda_crit - V[-1]
            result.notes.append(
                f"✗ DOES NOT cross critical. Gap to λ_crit: {gap:.4f} "
                f"({gap / self.cfg.lambda_crit * 100:.2f}% of λ_crit)"
            )

        # Physics analysis
        if self.cfg.include_rp3 and not crosses:
            result.notes.append(
                "RP³ curvature included but insufficient to reach λ_crit. "
                "Possible resolution: non-perturbative FRG flow beyond LPA, "
                "or V_UV larger than perturbative estimate at Planck scale."
            )
        if not self.cfg.include_rp3 and not crosses:
            result.notes.append("Flat spacetime: no curvature enhancement. V stays perturbative at all scales.")

        return result

    # ═══════════════════════════════════════════════════════════
    # Diagnostic: explore parameter space
    # ═══════════════════════════════════════════════════════════

    def find_critical_v_uv(self, tol: float = 1e-4) -> float | None:
        """Find the V_UV needed to exactly reach λ_crit at k_IR.

        This answers: how much larger must the UV coupling be
        for a spectral pole to form?
        """
        # Use bisection to find V_UV where V_IR = λ_crit
        lo, hi = self.cfg.v_uv, self.cfg.lambda_crit * 10
        orig = self.cfg.v_uv

        for _ in range(50):
            mid = (lo + hi) / 2
            self.cfg.v_uv = mid
            result = self.solve()
            if result.v_ir > self.cfg.lambda_crit:
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break

        self.cfg.v_uv = orig
        critical_v_uv = (lo + hi) / 2
        return critical_v_uv if critical_v_uv < self.cfg.lambda_crit * 5 else None


def analyze_flow(config: FlowConfig) -> FlowResult:
    """Convenience: run flow analysis and print summary."""
    solver = FRGFlowSolver(config)
    result = solver.solve()

    # Print summary
    print(f"\n{'=' * 64}")
    print(f"  FRG Flow Analysis — {config.operator_name}")
    print(f"{'=' * 64}")
    print(f"  k_UV = {config.k_uv:.4e} GeV   k_IR = {config.k_ir:.4e} GeV")
    print(f"  V_UV = {config.v_uv:.6e}       λ_crit = {config.lambda_crit:.2f}")
    print(f"  RP³ curvature: {'ON' if config.include_rp3 else 'OFF'}")
    print(f"  η_V feedback:  {'ON' if config.include_anomalous_dim else 'OFF'}")
    print(f"  V³ term:       {'ON' if config.include_v3 else 'OFF'}")
    print(f"  {'-' * 60}")
    print(f"  V(k_IR) = {result.v_ir:.6e}")
    print(f"  Log enhancement = {result.log_enhancement:.4f}")
    print(f"  Crosses λ_crit?  {result.crosses_critical}")
    if result.k_cross:
        print(f"    at k = {result.k_cross:.4e} GeV")
    print("\n  Beta function at IR:")
    print(f"  {'Component':<20} {'Value':>14} {'Fraction':>10}")
    print(f"  {'-' * 20} {'-' * 14} {'-' * 10}")
    for c in result.contributions:
        print(f"  {c['name']:<20} {c['value']:>14.6e} {c['fraction']:>10.4f}")
    print("\n  Notes:")
    for note in result.notes:
        print(f"    {note}")

    return result


def compute_all_flows() -> None:
    """Run FRG flow analysis for both protected operators."""
    # F²
    cfg_f2 = FlowConfig(
        operator_name="F² (SU(3) Gauge Field Strength)",
        v_uv=G3_MG**2 / (16.0 * np.pi**2),  # perturbative
        lambda_crit=28.0,
        include_rp3=True,
        include_anomalous_dim=True,
        include_v3=True,
    )
    result_f2 = analyze_flow(cfg_f2)

    # Tμν on RP³
    cfg_tmunu = FlowConfig(
        operator_name="Tμν on RP³ (L=2.44)",
        v_uv=(1.0 / L_RP3**4) / (16.0 * np.pi**2),  # curvature vertex
        lambda_crit=10.0,
        include_rp3=True,
        include_anomalous_dim=True,
        include_v3=True,
    )
    result_tmunu = analyze_flow(cfg_tmunu)

    # Flat spacetime comparison (Tμν vanishes by Ward identity)
    cfg_flat = FlowConfig(
        operator_name="Tμν (flat spacetime)",
        v_uv=(1.0 / L_RP3**4) / (16.0 * np.pi**2),
        lambda_crit=float("inf"),
        include_rp3=False,
        include_anomalous_dim=False,
        include_v3=False,
    )
    result_flat = analyze_flow(cfg_flat)

    return result_f2, result_tmunu, result_flat  # type: ignore[return-value]


if __name__ == "__main__":
    compute_all_flows()
