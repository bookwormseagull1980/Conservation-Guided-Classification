r"""Dyson-Schwinger Equation Solver for CGC Pole Formation.

# mypy: disable-error-code="arg-type, assignment, return-value"
Replaces the geometric-series "check V threshold" logic with a proper
self-consistent non-perturbative DSE solution on RP3 discrete modes.

All quantities are DIMENSIONLESS (normalized by appropriate powers of M_P).

Physics
-------
Gap equation:   x = V * sum_i w_i / (y_i + x)
Dressed Pi:     Pi(x) = sum_i w_i / (y_i + x)^2
Pole condition: V * Pi(x) >= 1

where:
  x = Sigma / M_P^2        (dimensionless gap)
  y_i = k_i^2 / M_P^2      (dimensionless eigenvalue)
  V is dimensionless vertex coupling
  w_i are dimensionless weights, normalized: sum w_i/y_i = Pi0_bare

The pole condition V * Pi(x) = 1 is the self-consistent version of
the geometric series: Pi_dressed = Pi_bare / (1 - V*Pi_bare)^2.

Author: CGC DSE Implementation
Date: 2026-07-29
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgc.rp3_engine.frg_flow_rp3 import (
    L_RP3,
    M_P,
    RP3Spectrum,
    f2_field_content,
    tmunu_field_content,
)
from cgc.rp3_engine.self_consistent_dyson import SelfConsistentSolver


@dataclass
class SpectralWeight:
    k_sq: float  # dimensionless y_i = k_i^2 / M_P^2
    weight: float  # dimensionless w_i
    species: str
    field_type: str
    degeneracy: int


def _build_mode_weights(channel: str = "Tmunu", L: float = L_RP3) -> tuple[list[SpectralWeight], float, float]:
    """Build DIMENSIONLESS mode weights.


# References
#     Dyson (1949): Dyson-Schwinger equations
#     Roberts & Williams (1994): Review of DSE formalism
#     Alkofer & von Smekal (2001): IR fixed points of QCD DSE
#

    Key: k_i^2 is divided by M_P^2, weights normalized such that
    sum(w_i / y_i) = SelfConsistentSolver.pi0_bare_ir (dimensionless).
    """
    spectrum = RP3Spectrum(L)
    if channel == "Tmunu":
        field_content = tmunu_field_content()
    elif channel == "F2":
        field_content = f2_field_content()
    else:
        raise ValueError(f"Unknown channel: {channel}")

    solver = SelfConsistentSolver(channel)
    target_pi0 = solver.pi0_bare_ir

    # Build raw mode list with dimensionless eigenvalues
    raw_entries = []
    k_max = 10.0 * M_P
    for fc in field_content:
        ft = fc.field_type
        rp3_modes = spectrum.all_modes_below(k_max, ft)
        for m in rp3_modes:
            dof = fc.n_species * fc.dof_per_species
            y = m.eigenvalue / M_P**2  # dimensionless
            raw_w = float(dof * m.degeneracy)
            if raw_w > 0 and y > 0:
                raw_entries.append((y, raw_w, m.degeneracy, fc.name, ft.value))

    raw_entries.sort(key=lambda e: e[0])

    y_arr = np.array([e[0] for e in raw_entries])
    w_raw = np.array([e[1] for e in raw_entries])

    raw_pi0 = float(np.sum(w_raw / y_arr))
    norm = target_pi0 / raw_pi0 if raw_pi0 > 0 else 1.0

    modes = []
    for i, (y, _wr, deg, name, ft) in enumerate(raw_entries):
        modes.append(
            SpectralWeight(
                k_sq=float(y),
                weight=float(w_raw[i] * norm),
                species=name,
                field_type=ft,
                degeneracy=deg,
            )
        )

    y_final = np.array([m.k_sq for m in modes])
    w_final = np.array([m.weight for m in modes])
    pi0_bare = float(np.sum(w_final / y_final))
    pi0_bubble = float(np.sum(w_final / y_final**2))

    return modes, pi0_bare, pi0_bubble


@dataclass
class DSEState:
    V: float
    x: float  # dimensionless gap x = Sigma / M_P^2
    Pi_bare: float
    Pi_dressed: float
    V_times_Pi: float  # V * Pi_dressed (>= 1 → pole)
    has_pole: bool
    converged: bool


@dataclass
class DSEResult:
    channel: str
    modes: list[SpectralWeight]
    Pi0_bare: float
    Pi0_bubble: float
    V_crit_tadpole: float
    V_crit_pole: float
    scan: list[DSEState]
    summary: dict


class DysonSchwingerSolver:
    """Self-consistent DSE gap equation solver on RP3.

    Uses dimensionless variables:
      y_i = k_i^2 / M_P^2
      x = Sigma / M_P^2
      gap eq:  x = V * sum w_i / (y_i + x)
    """

    def __init__(self, channel: str = "Tmunu", L: float = L_RP3):
        self.channel = channel
        self.L = L

        self.modes, self.Pi0_bare, self.Pi0_bubble = _build_mode_weights(channel, L)

        self._y = np.array([m.k_sq for m in self.modes])
        self._w = np.array([m.weight for m in self.modes])
        self._n = len(self._y)

        self.V_crit_tadpole = 1.0 / self.Pi0_bare if self.Pi0_bare > 0 else np.inf
        self.V_crit_bubble_bare = 1.0 / self.Pi0_bubble if self.Pi0_bubble > 0 else np.inf

        solver = SelfConsistentSolver(channel)
        self.V_native = solver.native_v

    def _S(self, x: float) -> float:
        """S(x) = sum w_i / (y_i + x), x ≥ 0 (physical condensate)."""
        xp = max(x, 0.0)
        return float(np.sum(self._w / (self._y + xp)))

    def _Pi(self, x: float) -> float:
        """Pi(x) = sum w_i / (y_i + x)^2, x ≥ 0."""
        d = self._y + max(x, 0.0)
        return float(np.sum(self._w / d**2))

    def solve_gap(self, V: float, max_iter: int = 200, tol: float = 1e-12) -> tuple[float, bool]:
        """Solve x = V * S(x). Returns (x, converged)."""
        if V <= 0:
            return 0.0, True
        if self.V_crit_tadpole * (1 - 1e-12) > V:
            return 0.0, True

        # Initial guess: the tadpole (x=0) value of the gap equation
        # x₀ = V·S(0) — the physical zero-field starting point of the
        # self-consistent iteration.
        x = V * self._S(0.0)

        for _i in range(max_iter):
            x_new = V * self._S(x)
            if not np.isfinite(x_new):
                # divergence (V at/near critical): report non-convergence
                return max(x, 0.0), False
            # under-relaxed update for numerical stability
            x_mix = 0.3 * x_new + 0.7 * x
            if abs(x_mix - x) / max(abs(x_mix), 1e-30) < tol:
                return max(x_mix, 0.0), True
            x = x_mix

        return max(x, 0.0), False

    def compute_state(self, V: float) -> DSEState:
        x, conv = self.solve_gap(V)
        Pi_d = self._Pi(x)
        VPi = V * Pi_d
        return DSEState(
            V=V, x=x, Pi_bare=self.Pi0_bare, Pi_dressed=Pi_d, V_times_Pi=VPi, has_pole=(VPi >= 1.0), converged=conv
        )

    def scan_V(self, V_min: float = 1e-6, V_max: float = None, n_pts: int = 200, log_scale: bool = True) -> DSEResult:
        if V_max is None:
            V_max = max(self.V_crit_tadpole * 10, self.V_crit_bubble_bare * 5, 1e4)

        V_vals = np.logspace(np.log10(V_min), np.log10(V_max), n_pts) if log_scale else np.linspace(V_min, V_max, n_pts)
        scan = [self.compute_state(V) for V in V_vals]

        V_crit_pole = np.inf
        for i in range(len(scan) - 1):
            s0, s1 = scan[i], scan[i + 1]
            if s0.has_pole != s1.has_pole:
                x0, x1 = s0.V_times_Pi, s1.V_times_Pi
                frac = (1.0 - x0) / (x1 - x0) if abs(x1 - x0) > 1e-15 else 0.5
                V_crit_pole = s0.V + frac * (s1.V - s0.V)
                break

        if V_crit_pole == np.inf:
            V_crit_pole = V_min if (scan and scan[0].has_pole) else np.inf

        summary = {
            "Pi0_bare": self.Pi0_bare,
            "Pi0_bubble": self.Pi0_bubble,
            "V_native": self.V_native,
            "V_crit_tadpole": self.V_crit_tadpole,
            "V_crit_bubble_bare": self.V_crit_bubble_bare,
            "V_crit_pole_dressed": V_crit_pole,
            "n_modes": self._n,
            "pole_formed": any(s.has_pole for s in scan),
            "k4_div_k2": self.Pi0_bubble / self.Pi0_bare if self.Pi0_bare > 0 else np.inf,
            "gap_tadpole": self.V_crit_tadpole / self.V_native if self.V_native > 0 else np.inf,
            "gap_bubble": self.V_crit_bubble_bare / self.V_native if self.V_native > 0 else np.inf,
        }
        return DSEResult(
            channel=self.channel,
            modes=self.modes,
            Pi0_bare=self.Pi0_bare,
            Pi0_bubble=self.Pi0_bubble,
            V_crit_tadpole=self.V_crit_tadpole,
            V_crit_pole=V_crit_pole,
            scan=scan,
            summary=summary,
        )

    def print_report(self, n_pts: int = 100) -> None:
        result = self.scan_V(n_pts=n_pts)
        s = result.summary

        print(f"\n{'=' * 70}")
        print(f"  DYSON-SCHWINGER EQUATION — {self.channel} Channel")
        print(f"{'=' * 70}")

        print("\n  Mode Space:")
        print(f"    RP3 modes: {self._n}")
        print(f"    y = k^2/M_P^2 range: {self._y.min():.4f} – {self._y.max():.2f}")

        print("\n  Bare Polarizations (dimensionless, y_i = k_i^2/M_P^2):")
        print(f"    Pi0 (1/y kernel):        {self.Pi0_bare:.6e}")
        print(f"    Pi0_bubble (1/y^2):      {self.Pi0_bubble:.6e}")
        print(f"    1/y^2 / 1/y ratio:       {s['k4_div_k2']:.6e}")

        print("\n  Critical Couplings:")
        print(f"    V_native:                  {self.V_native:.6e}")
        print(f"    V_crit_tadpole (gap):       {self.V_crit_tadpole:.4f}")
        print(f"    V_crit_bubble (bare pole):  {self.V_crit_bubble_bare:.4f}")
        print(f"    V_crit_dressed (DSE pole):  {s['V_crit_pole_dressed']:.4f}")

        print("\n  Enhancement Required (V_crit / V_native):")
        print(f"    Tadpole gap:   {s['gap_tadpole']:.1f}x")
        print(f"    Bubble pole:   {s['gap_bubble']:.1f}x")

        print("\n  Pole Formation:")
        if s["pole_formed"]:
            print(f"    YES — pole at V = {s['V_crit_pole_dressed']:.6e}")
        else:
            max_vpi = max(st.V_times_Pi for st in result.scan)
            print(f"    NO — max V*Pi = {max_vpi:.6e} < 1 in range")

        # Key comparison points
        print("\n  DSE vs Geometric Series:")
        print(f"    {'V':>12s}  {'GS V*Pi':>14s}  {'DSE V*Pi':>14s}  {'x=Sigma/M_P^2':>16s}")
        print(f"    {'─' * 12}  {'─' * 14}  {'─' * 14}  {'─' * 16}")
        for V in np.logspace(-6, min(np.log10(self.V_crit_tadpole * 5), 10), 8):
            state = self.compute_state(V)
            x_bare = V * self.Pi0_bare
            gs = V * self.Pi0_bare / max(1e-30, (1.0 - x_bare) ** 2) if x_bare < 1 else np.inf
            gs_s = f"{gs:.6e}" if gs < 1e8 else "          inf"
            print(f"    {V:12.4e}  {gs_s:>14s}  {state.V_times_Pi:14.6e}  {state.x:16.6e}")
        return result


def run_comparison() -> None:
    for channel in ["Tmunu", "F2"]:
        try:
            s = DysonSchwingerSolver(channel)
            s.print_report(n_pts=150)
        except Exception as e:
            print(f"\n  [{channel}] ERROR: {e}")
            import traceback

            traceback.print_exc()
    return 0


if __name__ == "__main__":
    run_comparison()
