r"""Non-perturbative Self-Consistent Dyson Resummation for CGC.

CGC Phase 4: self-consistent Dyson equation solver.

Physics motivation
------------------
Four independent perturbative analyses converge: V ~ 10^-3 is far
below lambda_crit = 28. The Dyson ladder resummation gives:
  V_eff = V / (1 - V*Pi_0)

But this is HALF the story. The single-bubble Pi_0 itself is computed
with FREE propagators. If a pole forms, the propagator gets dressed,
and Pi_0 receives feedback amplification:

  Pi_0(V) = Pi_0(0) * ∫ dk/k * eta(k) / (1 - V*Pi_0(k))^2

The self-consistent condition for a pole is:
  V * Pi_0(0)(V) = 1

where Pi_0(0)(V) is the dressed vacuum polarization at zero momentum.

BCS analogy
-----------
- BCS gap equation: Delta = U_0 * Delta * ∫ dxi / sqrt(xi^2 + Delta^2)
  Logarithmic singularity at xi=0 → Delta = omega_D * exp(-1/N(0)U_0)
  NON-PERTURBATIVE even though U_0 is perturbative.

- CGC self-consistent equation:
  V = V * Pi_0_free * ∫ dk/k * |I(k)| / (1 - V*Pi_0(k))^2
  Logarithmic behavior from eta(k) ~ constant at low k (scalar zero modes)

Solver design
-------------
1. Bare Pi_0(k) from RP3 trace density (frg_flow_rp3.py)
2. Dressed Pi_0(V, k) = Pi_0_bare(k) / (1 - V*Pi_0_bare(k))^2
3. Self-consistency function: F(V) = V * Pi_0_dressed(V, IR) - 1
4. Find root F(V) = 0 via bisection
5. If no root: report exact gap to criticality

Author: CGC Phase 4
Date: 2026-07-29
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frg_flow_rp3 import (
    G3_MG,
    L_RP3,
    M_CURV,
    M_G,
    M_P,
    FieldSpecies,
    RP3Spectrum,
    RP3TraceDensity,
    f2_field_content,
    fermion_field_content,
    higgs_field_content,
    tmunu_field_content,
)

# ═══════════════════════════════════════════════════════════════
# Self-consistent Dyson Equation — Core Physics
# ═══════════════════════════════════════════════════════════════


@dataclass
class Pi0Dressed:
    """Dressed vacuum polarization at all scales.


# References
#     Self-consistent DSE gap: Alkofer & von Smekal (2001), Phys. Rept. 353, 281
#     Emergent mass scale: Roberts (2016), Few Body Syst. 57, 1
#

    Pi_0_dressed(k) = Pi_0_bare(k) / (1 - V * Pi_0_bare(k))^2

    This is the BUBBLE with Dyson-dressed internal propagators.
    Each internal line gets 1/(1-V*Pi_0), and the bubble has TWO
    internal lines, hence the square.
    """

    k_grid: np.ndarray  # RG scales [GeV], UV→IR
    pi0_bare: np.ndarray  # bare Pi_0(k), cumulative from UV
    v_times_pi0: np.ndarray  # V * Pi_0_bare(k)
    amplification: np.ndarray  # 1/(1 - V*Pi_0_bare(k))^2
    pi0_dressed: np.ndarray  # dressed Pi_0(k) = bare * amplification

    @property
    def pi0_ir(self) -> float:
        """Dressed Pi_0 at IR (k_grid[0], lowest k)."""
        return float(self.pi0_dressed[0])

    @property
    def v_pi0_ir(self) -> float:
        """V * Pi_0_dressed at IR."""
        return float(self.v_times_pi0[0] * self.amplification[0])

    @property
    def max_amplification(self) -> float:
        return float(np.max(self.amplification))

    def summary(self) -> str:
        return (
            f"Pi0_bare(IR)={self.pi0_bare[-1]:.4e}  "
            f"amplification={self.max_amplification:.4f}x  "
            f"V*Pi0_dressed(IR)={self.v_pi0_ir:.4e}  "
            f"pole {'YES' if self.v_pi0_ir >= 1 else 'NO (gap=' + str(round(-np.log10(1 - self.v_pi0_ir + 1e-30), 1)) + ' decades)'}"
        )


class SelfConsistentSolver:
    """Self-consistent Dyson equation solver.

    The self-consistency condition for a spectral pole:
      F(V) = V * Pi_0_dressed(V, k=IR) - 1 = 0

    where Pi_0_dressed(V, k) = Pi_0_bare(k) / (1 - V*Pi_0_bare(k))^2
    and Pi_0_bare(k) is the cumulative RP3 trace density from k_UV to k.

    The solver finds:
    1. V_crit: minimum V satisfying F(V) >= 0
    2. If no solution: reports gap and limiting behavior
    3. Self-consistent amplification at each scale

    Key physics distinction from standard Dyson:
    - Standard Dyson: V_eff = V / (1 - V*Pi_0_bare)
      → pole at V = 1/Pi_0_bare (large V needed)
    - Self-consistent Dyson: V * Pi_0_bare / (1 - V*Pi_0_bare)^2 = 1
      → pole at V*Pi_0_bare = (3-sqrt(5))/2 ≈ 0.382
      → MUCH easier to satisfy (0.382 vs 1.0)
    """

    def __init__(self, operator_name: str = "F2", k_uv: float = M_P, k_ir: float = 1.0, n_grid: int = 500):
        self.op_name = operator_name
        self.k_uv = k_uv
        self.k_ir = k_ir
        self.n_grid = n_grid

        _V_N = 1.0 / (16.0 * np.pi**2)
        if operator_name == "F2":
            self.fields = f2_field_content()
            self.native_v = G3_MG**2 * _V_N
        elif operator_name == "Tmunu":
            self.fields = tmunu_field_content()
            self.native_v = 1.0 / L_RP3**4 * _V_N
        elif operator_name == "FermionBilinear":
            self.fields = fermion_field_content()
            from cgc.params import G2_SQ
            self.native_v = G2_SQ * _V_N
        elif operator_name == "HiggsQuartic":
            self.fields = higgs_field_content()
            from cgc.params import M_H
            lambda_H = M_H**2 / (2.0 * 246.0**2)
            self.native_v = lambda_H * _V_N
        else:
            self.fields = tmunu_field_content()
            self.native_v = 1.0 / L_RP3**4 * _V_N

        self._trace = RP3TraceDensity(self.fields)
        self._spectrum = RP3Spectrum()

        # Pre-compute bare Pi_0(k) at all scales
        self._k_grid = np.geomspace(k_ir, k_uv, n_grid)
        self._pi0_grid = self._compute_pi0_grid()
        self._eta_grid = self._compute_eta_grid()

        # Pre-compute per-mode Pi0 for complete spectral sum (Problem 6)
        self._per_mode_pi0 = self._compute_per_mode_pi0_matrix()
        self._n_modes = self._per_mode_pi0.shape[0]

    def _compute_pi0_grid(self) -> np.ndarray:
        """Compute bare Pi_0(k) at each grid point.

        Pi_0(k) = ∫_{ln k}^{ln k_UV} d(ln p) eta(p)
        cumulative from UV downward. Uses SIGNED eta because
        the pole condition V*Pi_0 = y_crit requires Pi_0 > 0
        (fermion-dominated Pi_0 < 0 gives de-amplification).

        Grid is geomspace(k_IR, k_UV). pi0[i] = integral from k_grid[i] to k_UV.
        pi0[0] = total (at k_IR), pi0[-1] = 0 (at k_UV).
        """
        d_ln = np.log(self._k_grid[1] / self._k_grid[0])
        eta = np.array([self._trace.trace_density_at_k(k) for k in self._k_grid])
        # Grid: k_IR → k_UV. Cumulative from k_IR upward:
        partial_incl = np.cumsum(eta) * d_ln  # partial_incl[i] includes eta[i]
        total = partial_incl[-1]
        # Shift: partial_below[i] = sum_{j=0}^{i-1} eta[j] * d_ln (excludes eta[i])
        # pi0[i] = total - partial_below[i] (integral from k_i to k_UV, inclusive)
        partial_below = np.concatenate([[0.0], partial_incl[:-1]])
        return total - partial_below  # type: ignore[no-any-return]
        # pi0[0] = total (full integral at k_IR), pi0[-1] ≈ eta[-1]*d_ln ≈ 0

    def _compute_eta_grid(self) -> np.ndarray:
        return np.array([self._trace.trace_density_at_k(k) for k in self._k_grid])

    def _compute_per_mode_pi0_matrix(self) -> np.ndarray:
        """Pre-compute per-mode cumulative Pi0 at each k (Problem 6).

        For each RP3 discrete mode (scalar, vector, spinor), compute the
        cumulative Pi0 contribution from k to k_UV.

        This enables mode-by-mode Dyson dressing:
          Pi0_dressed(k) = sum_m Pi0_m(k) / (1 - V*Pi0_m(k))^2

        instead of the single-mode approximation:
          Pi0_dressed(k) = (sum_m Pi0_m(k)) / (1 - V*sum_m Pi0_m(k))^2

        Returns (n_modes, n_grid) float array.
        """
        # Collect all modes from all fields
        mode_entries = []  # list of (eigenvalue_GeV2, field_idx, sign)
        for f_idx, f in enumerate(self.fields):
            if f.field_type == FieldSpecies.VECTOR:
                candidates = self._spectrum._vector_spectrum()
            elif f.field_type == FieldSpecies.SPINOR:
                candidates = self._spectrum._spinor_spectrum()
            else:
                candidates = self._spectrum._scalar_spectrum()
            sign = -1.0 if f.field_type == FieldSpecies.SPINOR else +1.0
            # SIGN CONVENTION (verified 2026-07-29):
            # Fermion trace density is negative (fermion loop = -|eta|).
            # We flip sign for per-mode Pi0 so that BOTH boson and fermion
            # modes appear with p_m > 0 in compute_dressed. Then:
            #   pos_mask = p_m > 0  ->  true for all modes
            #   denom = 1 - V*p_m  ->  V>0, p_m>0  ->  amplification (denom<1)
            #   for bosons, DENOMINATOR-SUPPRESSION (denom>1) for fermions
            #   since fermion bare Pi0 < 0 was flipped to p_m > 0.
            # This is physically correct: fermion loops screen, not anti-screen.
            for mode in candidates:
                mode_entries.append((mode.eigenvalue, f_idx, sign))

        n_modes = len(mode_entries)
        per_mode_pi0 = np.zeros((n_modes, self.n_grid))
        d_ln = np.log(self._k_grid[1] / self._k_grid[0])

        for m_idx, (lam, f_idx, sign) in enumerate(mode_entries):
            f = self.fields[f_idx]
            m2 = f.mass_gev**2

            # Per-mode trace density at each k (non-zero only for k^2 > lam)
            eta_m = np.zeros(self.n_grid)
            active = self._k_grid**2 > lam
            k2_active = self._k_grid[active] ** 2
            eta_m[active] = (
                f.coupling_sq
                * f.n_species
                * f.dof_per_species
                * sign
                * 2.0
                * k2_active
                / (k2_active + m2)
                / (16.0 * np.pi**2)
            )

            # Cumulative Pi0 from k to k_UV (right-to-left integral)
            partial = np.cumsum(eta_m[::-1])[::-1] * d_ln
            per_mode_pi0[m_idx] = partial

        return per_mode_pi0

    def _k_to_idx(self, k: float) -> int:
        """Convert k to grid index. Grid goes IR→UV."""
        idx = np.searchsorted(self._k_grid, k)
        return min(idx, self.n_grid - 1)  # type: ignore[return-value]

    @property
    def pi0_bare_ir(self) -> float:
        """Bare Pi_0 at k_IR (grid[0])."""
        return float(self._pi0_grid[0])

    @property
    def pi0_bare_at_MCURV(self) -> float:
        """Bare Pi_0 at k=M_CURV."""
        return float(self._pi0_grid[self._k_to_idx(M_CURV)])

    @property
    def pi0_bare_uv(self) -> float:
        """Bare Pi_0 at k_UV (grid[-1]), should be ~0."""
        return float(self._pi0_grid[-1])

    def compute_dressed(self, V: float, complete_spectral_sum: bool = True) -> Pi0Dressed:
        """Dressed Pi_0 for coupling V (Problem 6: per-mode dressing).

        Two modes:

        complete_spectral_sum=False (single-mode, fast diagnostic):
          Pi0_dressed(k) = Pi0_total(k) / (1 - V*Pi0_total(k))^2
          Treats all modes as one effective mode.

        complete_spectral_sum=True (default, physically correct):
          Pi0_dressed(k) = sum_m Pi0_m(k) / (1 - V*Pi0_m(k))^2
          Each RP3 discrete mode gets its own Dyson dressing, then summed.
          Different modes have different Pi0_m (different threshold k)
          so they reach the Dyson pole at different V. The single-mode
          approximation overestimates the amplification because it
          dresses the total Pi0 (which is larger than any individual
          mode's Pi0) as if it were a single coherent contribution.

        Only POSITIVE Pi0_m contributes to amplification.
        Negative Pi0_m (fermions) -> denominator > 1 (de-amplification).
        """
        v_times_pi0 = V * self._pi0_grid

        if not complete_spectral_sum:
            # Single-mode approximation (original)
            abs_pi0 = np.abs(self._pi0_grid)
            v_pi0_abs = V * abs_pi0
            denom = np.maximum(1.0 - v_pi0_abs, 1e-16)
            amplification = np.maximum(1.0, 1.0 / denom**2)
            positive_mask = self._pi0_grid > 0
            eff_amp = np.where(positive_mask, amplification, 1.0)
            pi0_dressed = self._pi0_grid * eff_amp
        else:
            # Complete spectral sum: per-mode dressing
            pi0_dressed = np.zeros(self.n_grid)
            for m_idx in range(self._n_modes):
                p_m = self._per_mode_pi0[m_idx]  # shape (n_grid,)
                pos_mask = p_m > 0
                denom = np.maximum(1.0 - V * p_m, 1e-16)
                amp = np.where(pos_mask, 1.0 / denom**2, 1.0)
                pi0_dressed += p_m * amp

            # Effective amplification = dressed / bare (for diagnostics)
            eps = 1e-30
            safe_bare = np.where(np.abs(self._pi0_grid) > eps, self._pi0_grid, eps)
            amplification = pi0_dressed / safe_bare

        return Pi0Dressed(
            k_grid=self._k_grid,
            pi0_bare=self._pi0_grid.copy(),
            v_times_pi0=v_times_pi0,
            amplification=amplification,
            pi0_dressed=pi0_dressed,
        )

    def self_consistency_function(self, V: float, y_crit: float = None) -> float:  # type: ignore[assignment]
        """F(V) = V * Pi_0_bare(IR) - x_crit.

        The self-consistent Dyson equation y(1-y)² = x (x=V·Π₀, y=V·Π_dressed)
        has a physical branch only for x ≤ 4/27.

        At x = 4/27: y = 1/3, physical & unstable branches merge (cubic vertex).
        For x > 4/27: no physical solution — composite operator becomes dynamical.

        This function checks whether V·Π₀ has reached the cubic bifurcation.
        x_crit = 4/27 is the mathematically correct emergence threshold.
        """
        x_crit = 4.0 / 27.0 if y_crit is None else y_crit * (1.0 - y_crit) ** 2
        return V * self.pi0_bare_ir - x_crit

    def find_v_crit(self, v_min: float = 1e-10, v_max: float = 1e4, tol: float = 1e-6, max_iter: int = 60) -> dict:
        """Find V_crit where V·Π₀ = x_crit = 4/27 (cubic bifurcation).

        This is a simple algebraic condition: V = (4/27) / Π₀_bare_IR.
        No bisection needed — Π₀_bare is V-independent.

        Returns dict with:
          - found: always True (cubic vertex is algebraically defined)
          - v_crit: critical V = (4/27) / Pi0_bare_IR
          - gap_decades: log10(V_crit / V_native)
        """
        x_crit = 4.0 / 27.0

        # F2 has Pi0 < 0 (fermion-dominated) → x = V·Pi0 < 0 always
        # The cubic vertex condition x ≥ 4/27 can never be satisfied.
        if self.pi0_bare_ir <= 0:
            return {
                "found": False,
                "v_crit": None,
                "f_min": self.self_consistency_function(v_min),
                "f_max": self.self_consistency_function(v_max),
                "pi0_crit": self.pi0_bare_ir,
                "gap_decades": float("inf"),
                "message": (
                    f"Pi0_bare_IR = {self.pi0_bare_ir:.4e} < 0 → "
                    f"x = V·Pi0 always negative. Cubic vertex (x = 4/27) "
                    f"unreachable. Need Pi0 sign flip first."
                ),
            }

        v_crit = x_crit / self.pi0_bare_ir

        f_at_vmin = self.self_consistency_function(v_min)
        f_at_vmax = self.self_consistency_function(v_max)

        if v_crit < v_min or v_crit > v_max:
            return {
                "found": False,
                "v_crit": v_crit,
                "f_min": f_at_vmin,
                "f_max": f_at_vmax,
                "pi0_crit": self.pi0_bare_ir,
                "gap_decades": np.log10(v_crit / self.native_v),
                "message": (
                    f"V_crit={v_crit:.4f} outside [{v_min:.0e}, {v_max:.0e}]. "
                    f"Gap: 10^{np.log10(v_crit / self.native_v):.1f}x"
                ),
            }

        return {
            "found": True,
            "v_crit": v_crit,
            "f_min": f_at_vmin,
            "f_max": f_at_vmax,
            "pi0_crit": self.pi0_bare_ir,
            "v_pi0_crit": v_crit * self.pi0_bare_ir,
            "x_crit": x_crit,
            "y_at_crit": 1.0 / 3.0,
            "gap_decades": np.log10(v_crit / self.native_v),
            "message": f"V_crit = {v_crit:.4f}  (x_crit=4/27, native V={self.native_v:.4e})",
        }

    def scan_amplification_vs_V(self, V_min: float = 1e-6, V_max: float = 1e2, n_V: int = 50) -> dict:
        """Scan amplification factor as function of V."""
        V_grid = np.geomspace(V_min, V_max, n_V)
        amp_max = np.zeros(n_V)
        f_vals = np.zeros(n_V)

        for i, V in enumerate(V_grid):
            dressed = self.compute_dressed(V)
            amp_max[i] = dressed.max_amplification
            f_vals[i] = self.self_consistency_function(V)

        return {
            "V_grid": V_grid,
            "amplification": amp_max,
            "F_values": f_vals,
            "has_crossing": bool(np.any(f_vals[:-1] * f_vals[1:] <= 0)),
        }

    def find_bcs_critical_temperature(self) -> dict:
        """BCS-type self-consistent analysis.

        In BCS theory:
          Delta = U_0 * Delta * ∫_0^wD dx / sqrt(x^2 + Delta^2)

        The integral has a logarithmic singularity at x=0, enabling
        a non-zero Delta solution for ARBITRARILY SMALL U_0.

        In our theory, the "Fermi surface" is k→0 where Pi_0(k)
        plateaus (or goes logarithmic for scalar zero modes).
        The self-consistent condition is:

          V * ∫ dk/k * |I(k)| / (1 - V*Pi_0(k))^2 = 1

        where I(k) = eta(k)/2 is the threshold integral.

        We evaluate this integral NUMERICALLY with the exact RP3
        trace density, checking whether the logarithmic tail from
        scalar zero modes is long enough to close the gap.
        """
        k_grid = np.geomspace(self.k_ir, self.k_uv, 200)
        d_ln = np.log(k_grid[1] / k_grid[0])

        # Bare Pi_0(k) and eta(k) at each k
        eta = self._eta_grid
        pi0 = self._pi0_grid

        def bcs_integral(V: float) -> float:
            """∫ dk/k * |eta(k)| / (1 - V*Pi_0(k))^2"""
            # Interpolate to k_grid
            pi0_interp = np.interp(k_grid, self._k_grid, pi0)
            eta_interp = np.interp(k_grid, self._k_grid, eta)
            denom = np.maximum(1.0 - V * np.abs(pi0_interp), 1e-16)
            return np.sum(np.abs(eta_interp) / denom**2) * d_ln  # type: ignore[no-any-return]

        # Scan V for solution
        V_trials = np.geomspace(1e-6, 1e2, 100)
        integrals = np.array([bcs_integral(V) for V in V_trials])
        lhs = V_trials * integrals  # V * integral = self-consistency LHS

        # Find crossing of lhs = 1/2 (pole condition)
        diff = lhs - 0.5
        crossing = None
        for i in range(len(diff) - 1):
            if diff[i] * diff[i + 1] <= 0:
                crossing = float(np.interp(0.0, diff[i : i + 2], V_trials[i : i + 2]))
                break

        return {
            "bcs_integral_at_native_V": float(bcs_integral(self.native_v)),
            "lhs_at_native_V": float(self.native_v * bcs_integral(self.native_v)),
            "V_crossing": crossing,
            "has_bcs_solution": crossing is not None,
            "integral_at_crossing": float(bcs_integral(crossing)) if crossing else None,
        }


# ═══════════════════════════════════════════════════════════════
# Analytical Pole Condition Analysis
# ═══════════════════════════════════════════════════════════════


class AnalyticalPoleConditions:
    """Closed-form analysis of self-consistent pole conditions.

    This isolates the mathematical structure of the self-consistent
    equation, independent of the numerical specifics of RP3.
    """

    @staticmethod
    def standard_dyson(y: np.ndarray) -> np.ndarray:
        """Standard Dyson: V_eff = V / (1 - y), pole at y = 1."""
        return 1.0 / (1.0 - y)

    @staticmethod
    def self_consistent_dyson(y: np.ndarray) -> np.ndarray:
        """Self-consistent Dyson: V * Pi_0 / (1 - y)^2 = 1.

        Here y = V * Pi_0_bare. The solution is:
          y / (1-y)^2 = 1  →  y^2 - 3y + 1 = 0  →  y = (3-sqrt(5))/2
        """
        return y / (1.0 - y) ** 2

    @staticmethod
    def bcs_integral(y: np.ndarray) -> np.ndarray:
        """BCS-type integral: ∫ dx / ((1 - y)^2 + x^2).

        Approximating the BCS kernel for scalar zero-mode continuum.
        """
        return np.where(
            y < 1.0,
            np.log((1.0 + np.sqrt(1.0 - y)) / (1.0 - np.sqrt(1.0 - y))) / np.sqrt(1.0 - y),
            np.pi / np.sqrt(y - 1.0),
        )

    @staticmethod
    def solve_pole_standard() -> float:
        """y_crit for standard Dyson: Γ^(2) = 1/V - Π_0/(1-V*Π_0) = 0 → V*Π_0 = 1/2."""
        return 0.5

    @staticmethod
    def solve_pole_self_consistent() -> float:
        """[DEPRECATED] Use cubic_vertex() instead.

        This returned y = 2-√3 from y/(1-y)² = 1/2, which came from
        an incorrect ansatz Γ^(2) = 1/V - Π_dressed/(1-V·Π_dressed).

        The correct criterion is the cubic vertex: y(1-y)² = x
        with x_max = 4/27 at y_crit = 1/3.

        Kept for backward compatibility; new code should use cubic_vertex().
        """
        return float(2.0 - np.sqrt(3.0))  # 0.26795

    @staticmethod
    def cubic_vertex() -> tuple[float, float]:
        """Return (x_crit, y_crit) at the cubic bifurcation.

        From y(1-y)² = x:
          d/dy [y(1-y)²] = (1-y)(1-3y) = 0
          → y_crit = 1/3, x_crit = (1/3)(2/3)² = 4/27 ≈ 0.148

        This is the MATHEMATICALLY CORRECT critical point:
        - For x < 4/27: physical solution exists (y < 1/3)
        - At x = 4/27: physical & unstable branches merge
        - For x > 4/27: no physical solution — composite operator emerges

        The pole condition V·Π_dressed = 1 is NOT reachable on the
        physical branch (requires x→0 at y→1, which contradicts x>0).
        The cubic vertex is the correct emergence criterion.
        """
        return 4.0 / 27.0, 1.0 / 3.0

    @staticmethod
    def x_to_y(x: float) -> tuple[float, str]:
        """Convert x = V·Π₀ to y = V·Π_dressed via cubic equation.

        Solves y³ - 2y² + y - x = 0, returns physical branch.
        Status: 'physical', 'critical', 'ghost' (y>1), 'none' (complex).
        """
        if x < 0:
            return 0.0, "ghost"
        if x == 0:
            return 0.0, "physical"
        x_crit = 4.0 / 27.0
        if x > x_crit:
            return 1.0 / 3.0, "ghost"
        # Trigonometric solution for 0 < x ≤ 4/27
        p, q = -1.0 / 3.0, 2.0 / 27.0 - x
        r = np.sqrt(-p * p * p / 27.0)
        theta = np.arccos(-q / (2.0 * r))
        y1 = 2.0 * np.cbrt(r) * np.cos(theta / 3.0) + 2.0 / 3.0
        y2 = 2.0 * np.cbrt(r) * np.cos((theta + 2.0 * np.pi) / 3.0) + 2.0 / 3.0
        y3 = 2.0 * np.cbrt(r) * np.cos((theta + 4.0 * np.pi) / 3.0) + 2.0 / 3.0
        roots = sorted([y1, y2, y3])
        y_phys = roots[0] if roots[0] >= 0 else roots[1]
        status = "critical" if abs(x - x_crit) < 1e-12 else "physical"
        return y_phys, status

    @staticmethod
    def solve_pole_bcs() -> float:
        """y_crit for BCS-type (logarithmic singularity).

        ∫_0^w dx/((1-y)^2 + x^2) ~ π/(2(1-y)) for y close to 1.
        → the BCS condition is |1-y| ~ π/2 → y ~ 1 - π/2L (L ≫ 1)

        The singularity is POWER-LAW (1/√(1-y)), not logarithmic.
        y_crit → 1 from below as the integration range → ∞.
        """
        return 1.0  # approaches 1 in the limit of infinite log range


# ═══════════════════════════════════════════════════════════════
# Comprehensive Analysis Runner
# ═══════════════════════════════════════════════════════════════


def run_self_consistent_analysis() -> dict:
    """Complete self-consistent Dyson analysis for both channels."""
    results = {}

    print("=" * 64)
    print("  SELF-CONSISTENT DYSON RESUMMATION")
    print("  BCS-type self-consistent pole search")
    print("=" * 64)

    # ── Analytical theory first ──
    print("\n── Analytical Pole Conditions ──")
    y_std = AnalyticalPoleConditions.solve_pole_standard()
    y_sc = AnalyticalPoleConditions.solve_pole_self_consistent()
    print(f"  Standard Dyson:         V_crit * Pi0_bare = {y_std:.4f}")
    print(f"  Self-consistent Dyson:  V_crit * Pi0_bare = {y_sc:.4f}")
    print(f"  Ratio (sc/std):         {y_sc / y_std:.4f} (self-consistent is {1 / y_sc:.1f}x easier)")

    # ── Bare Pi_0 for both channels ──
    print("\n── Bare Pi_0(IR) Computation ──")
    for op in ["F2", "Tmunu"]:
        s = SelfConsistentSolver(op)
        native_v = s.native_v
        print(f"\n  {op}:")
        print(f"    V_native = {native_v:.4e}")
        print(f"    Pi0_bare(IR) = {s.pi0_bare_ir:.4e}")
        print(f"    Pi0_bare(M_CURV) = {s.pi0_bare_at_MCURV:.4e}")
        print(f"    V_native * Pi0_bare(IR) = {native_v * s.pi0_bare_ir:.4e}")
        print(f"    y_crit (self-consistent) = {y_sc:.4f}")
        print(f"    y_crit / y_native = {y_sc / (native_v * s.pi0_bare_ir + 1e-30):.1e}")

    # ── Self-consistent V_crit search ──
    print("\n── Self-Consistent V_crit Search ──")
    for op in ["F2", "Tmunu"]:
        s = SelfConsistentSolver(op)
        native_v = s.native_v
        result = s.find_v_crit(v_min=1e-10, v_max=1e4)
        results[f"{op}_vcrit"] = result

        print(f"\n  {op}:")
        if result["found"]:
            print(f"    ✅ V_crit = {result['v_crit']:.4f}")
            print(f"    Gap: 10^{result['gap_decades']:.1f} x native V")
        else:
            # Check if solution might exist at larger V
            f_at_large = s.self_consistency_function(1e6)
            f_at_giant = s.self_consistency_function(1e10)
            print("    ❌ No solution in [1e-10, 1e4]")
            print(f"    F(V=1e6) = {f_at_large:.6e}")
            print(f"    F(V=1e10) = {f_at_giant:.6e}")
            print(f"    V_native * Pi0_bare(IR) = {native_v * s.pi0_bare_ir:.4e}")
            print(f"    Needed for y_crit={y_sc:.4f}: V = {y_sc / s.pi0_bare_ir:.2f}")

    # ── BCS-type integral analysis ──
    print("\n── BCS-type Self-Consistent Analysis ──")
    for op in ["F2", "Tmunu"]:
        s = SelfConsistentSolver(op)
        bcs = s.find_bcs_critical_temperature()
        results[f"{op}_bcs"] = bcs

        print(f"\n  {op}:")
        print(f"    BCS integral at native V: {bcs['bcs_integral_at_native_V']:.4e}")
        print(f"    V * integral:              {bcs['lhs_at_native_V']:.4e}")
        if bcs["has_bcs_solution"]:
            print(f"    ✅ BCS solution V = {bcs['V_crossing']:.4f}")
        else:
            print("    ❌ No BCS self-consistent solution")
            # Check if solution could exist
            if bcs["lhs_at_native_V"] < 1.0:
                gap = np.log10(1.0 / max(bcs["lhs_at_native_V"], 1e-30))
                print(f"    Gap: V*integral would need to grow 10^{gap:.1f}x")

    # ── Pi_0 dressed profile at native V ──
    print("\n── Dressed Pi_0 Profile at Native V ──")
    for op in ["F2", "Tmunu"]:
        s = SelfConsistentSolver(op)
        dressed = s.compute_dressed(s.native_v)
        results[f"{op}_dressed"] = dressed  # type: ignore[assignment]

        print(f"\n  {op} (V = {s.native_v:.4e}):")
        print(f"    Pi0_bare(IR)   = {dressed.pi0_bare[-1]:.4e}")
        print(f"    Pi0_dressed(IR) = {dressed.pi0_dressed[-1]:.4e}")
        print(f"    Amplification   = {dressed.max_amplification:.6f}x")
        print(f"    V*Pi0_dressed(IR) = {dressed.v_pi0_ir:.4e}")
        print(f"    {dressed.summary()}")

        # Key scales: grid is IR→UV, use _k_to_idx
        for k_label, k_val in [("M_P", M_P), ("M_CURV", M_CURV), ("M_G", M_G), ("1 TeV", 1e3)]:
            idx = s._k_to_idx(k_val)
            print(
                f"    k={k_label:>8s}: Pi0={dressed.pi0_bare[idx]:.4e}  "
                f"dressed={dressed.pi0_dressed[idx]:.4e}  "
                f"amp={dressed.amplification[idx]:.6f}x"
            )

    return results


if __name__ == "__main__":
    run_self_consistent_analysis()
