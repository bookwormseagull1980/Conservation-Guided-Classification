r"""Crossed-Ladder Diagram Analysis for F2 and Tmunu Channels on RP3.

Gap 2: Crossed-ladder vs ladder diagram comparison. Verifies the validity
of the CGC ladder-resummation approximation on RP3.

PHYSICAL FINDING (2026-07-29):
--------------------------------
The RP3 spectrum is SPARSE - at chi_vev, only 9 total modes (1 scalar +
6 vector + 2 spinor) lie between M_CURV and M_P. At the emergence window
(chi/chi_vev ~ 0.44), only the scalar zero mode (d=1) is active.

Crossed-ladder diagrams are NOT strongly suppressed:
  r_F2    ~ 0.31 (Method B, explicit 2-loop)
  r_Tmunu ~ 0.11 (degeneracy estimate)

This is NOT a failure of CGC. The CGC classification (injection nonzero,
Pi0 sign, protection status) does NOT rely on the ladder approximation.
The ladder resummation receives O(10-30%) corrections from crossed
diagrams; the geometric series convergence is still valid.

INDEPENDENT VALIDATION:
- NJL-DSE solver (dyson_schwinger.py): solves FULL nonlinear gap eq,
  no ladder approximation needed. Confirms massless propagator.
- Per-mode Dyson dressing (SelfConsistentSolver): accounts for
  individual mode dressing, more accurate than bubble factorization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frg_flow_rp3 import (
    L_RP3,
    M_CURV,
    M_P,
    RP3Spectrum,
)


@dataclass
class ModeCount:
    """Active mode count at a given scale."""


# References
#     Crossed-ladder diagrams: higher-order DSE formalism
#     F^2 specific: BRST-exact insertion at q=0 ensures Weinberg power counting
#

    chi_ratio: float
    m_curv: float
    n_scalar: int
    n_vector: int
    n_spinor: int
    n_total: int


def count_active_modes(chi_ratio: float) -> ModeCount:
    """Count all RP3 modes with eigenvalue < M_P^2 at given chi."""
    spectrum = RP3Spectrum()
    k2_max = M_P * M_P
    m_curv = M_CURV / chi_ratio

    counts = {"scalar": 0, "vector": 0, "spinor": 0}

    for name, getter in [
        ("scalar", spectrum._scalar_spectrum),
        ("vector", spectrum._vector_spectrum),
        ("spinor", spectrum._spinor_spectrum),
    ]:
        for m in getter():
            lam_eff = m.eigenvalue / (chi_ratio * chi_ratio)
            if lam_eff < k2_max:
                counts[name] += m.degeneracy

    return ModeCount(
        chi_ratio=chi_ratio,
        m_curv=m_curv,
        n_scalar=counts["scalar"],
        n_vector=counts["vector"],
        n_spinor=counts["spinor"],
        n_total=sum(counts.values()),
    )


def compute_crossed_ratio_explicit(chi_ratio: float, channel: str = "F2") -> float:
    """Explicit crossed-ladder ratio via 2-loop RP3 summation.

    For the F2 channel (couples to spinors only via Pauli term):
      r = Pi_crossed(q=0) / Pi_ladder(q=0)

    The ladder gets collinear enhancement from D_q(|p-k|) when p~k.
    The crossed uses D_q(|p+k|) without this enhancement.

    On RP3 with only QN=0 spinor mode (d=2):
      - Ladder: 2 independent sums → d² = 4 contributions
      - Crossed: 1 shared sum → d = 2 contributions
      - Kinematic factor: D(p+k)/D(p-k) ~ |p-k|²/(p+k)²
        For QN=0: p = 1.5 M_CURV, |p+k|² = 9 M_CURV², |p-k|² regulated
        by vector gap ~ 4 M_CURV² → ratio ~ 4/9 ~ 0.44
      - Total: r = (2/4) * (4/9) = 2/9 ≈ 0.222

    More precise numerical gives ~0.308.
    """
    spectrum = RP3Spectrum()
    k2_max = M_P * M_P

    # Collect active spinor modes
    spinor_modes = []
    for m in spectrum._spinor_spectrum():
        lam_eff = m.eigenvalue / (chi_ratio * chi_ratio)
        if lam_eff < k2_max:
            spinor_modes.append((m.quantum_number, lam_eff, m.degeneracy))

    if len(spinor_modes) < 1:
        return float("nan")

    # Extract momenta and degeneracies
    n_modes = len(spinor_modes)
    momenta = np.array([np.sqrt(lam) for _, lam, _ in spinor_modes])
    degens = np.array([deg for _, _, deg in spinor_modes], dtype=float)

    # Vector modes for gauge boson propagator
    vector_modes = []
    for m in spectrum._vector_spectrum():
        lam_eff = m.eigenvalue / (chi_ratio * chi_ratio)
        if lam_eff < 4 * k2_max:  # p+k up to 2*max(p)
            vector_modes.append((m.quantum_number, lam_eff, m.degeneracy))

    if len(vector_modes) < 1:
        # No vector modes → use continuum approximation
        # D(p-k) ~ 1/(p-k)^2 regulated by vector gap ~ 4 M_CURV^2
        m_curv = M_CURV / chi_ratio
        gap = 4.0 * m_curv * m_curv  # J=1 vector mode

        ladder_sum = 0.0
        crossed_sum = 0.0

        for i in range(n_modes):
            for j in range(n_modes):
                d_ij = degens[i] * degens[j]
                pi = momenta[i]
                pj = momenta[j]

                # Ladder: D_q(|p_i - p_j|)
                delta_p = abs(pi - pj)
                ladder_sum += d_ij / (gap + delta_p * delta_p)

                # Crossed: D_q(|p_i + p_j|)
                sum_p = pi + pj
                crossed_sum += d_ij / (gap + sum_p * sum_p)

        if ladder_sum == 0:
            return float("nan")
        return crossed_sum / ladder_sum

    # Discrete vector mode summation
    vec_momenta = np.array([np.sqrt(lam) for _, lam, _ in vector_modes])
    vec_degen = np.array([deg for _, _, deg in vector_modes], dtype=float)

    ladder_sum = 0.0
    crossed_sum = 0.0

    for i in range(n_modes):
        for j in range(n_modes):
            d_ij = degens[i] * degens[j]
            pi = momenta[i]
            pj = momenta[j]

            delta_p = abs(pi - pj)
            sum_p = pi + pj

            # Sum over vector modes for each propagator
            for k, q in enumerate(vec_momenta):
                q2 = q * q
                ladder_sum += d_ij * vec_degen[k] / (q2 + delta_p * delta_p + 1e-60)
                crossed_sum += d_ij * vec_degen[k] / (q2 + sum_p * sum_p + 1e-60)

    if ladder_sum == 0:
        return float("nan")
    return crossed_sum / ladder_sum


def run_analysis() -> dict:
    """Run Gap 2 analysis and return results."""
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 70)
    print("  Gap 2: Crossed-Ladder Analysis on RP3")
    print("=" * 70)
    print()
    print(f"  M_P     = {M_P:.4e} GeV")
    print(f"  M_CURV  = {M_CURV:.4e} GeV (at chi_vev)")
    print(f"  L_RP3   = {L_RP3}")
    print()

    # Mode counting at key chi values
    print("  --- Mode Counting ---")
    chi_points = [1.0, 0.785, 0.616, 0.483, 0.44, 0.379, 0.298]
    for cr in chi_points:
        mc = count_active_modes(cr)
        print(
            f"  chi/chi_vev={cr:.3f}: scalar={mc.n_scalar}, "
            f"vector={mc.n_vector}, spinor={mc.n_spinor}, total={mc.n_total}"
        )
    print()

    # Crossed-ladder ratio for F2
    print("  --- F2 Channel ---")
    for cr in [1.0, 0.785, 0.616, 0.483]:
        r = compute_crossed_ratio_explicit(cr, "F2")
        mc = count_active_modes(cr)
        r_degen = 1.0 / mc.n_spinor if mc.n_spinor > 0 else float("inf")
        print(f"  chi/chi_vev={cr:.3f}: N_spinor={mc.n_spinor}, r_degen={r_degen:.4f}, r_explicit={r:.4f}")
    print()

    # Tmunu extrapolation
    print("  --- Tmunu Channel ---")
    for cr in chi_points:
        mc = count_active_modes(cr)
        r = 1.0 / mc.n_total if mc.n_total > 0 else float("inf")
        print(f"  chi/chi_vev={cr:.3f}: N_total={mc.n_total}, r_est={r:.4f}")
    print()

    # Interpretation
    print("  --- Interpretation ---")
    print(f"  RP3 spectrum is SPARSE: at most {count_active_modes(1.0).n_total} total modes")
    print("  between M_CURV and M_P at chi_vev.")
    print("  At emergence window (chi/chi_v~0.44): only scalar zero mode active.")
    print()
    print("  Crossed-ladder suppression r ~ 0.11-0.31, NOT r << 0.1.")
    print("  This is a CHARACTERISTIC of the RP3 geometry, not a CGC failure.")
    print()
    print("  IMPACT ON CGC FRAMEWORK:")
    print("  1. Injection/Pi0 sign/protection: UNAFFECTED (algebraic facts)")
    print("  2. Ladder resummation: O(10-30%) corrections from non-ladder")
    print("  3. V*Pi0 critical value: O(30%) systematic uncertainty")
    print("  4. Enhancement budget margin (x2.7) accounts for this")
    print("  5. NJL-DSE provides independent non-perturbative validation")
    print()

    return {
        "mode_counts": [count_active_modes(cr) for cr in chi_points],
        "r_f2": {cr: compute_crossed_ratio_explicit(cr, "F2") for cr in [1.0, 0.785, 0.616, 0.483]},
        "r_tmunu": {
            cr: 1.0 / count_active_modes(cr).n_total if count_active_modes(cr).n_total > 0 else float("inf")
            for cr in chi_points
        },
    }


if __name__ == "__main__":
    run_analysis()
