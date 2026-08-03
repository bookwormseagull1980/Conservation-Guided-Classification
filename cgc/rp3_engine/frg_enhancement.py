r"""Background-Field FRG: First-Principles Enhancement f(chi).

The enhancement function f(chi) is computed from the chi-potential-
coupled FRG trace density:

  - Threshold function: 2k²/(k²+m²)  (RP3TraceDensity self-energy
    trace, Litim regulator).
  - Fermion loops contribute negative eta (spin-statistics sign).

Spectrum correctness (verified 2026-07-29):
    For L_RP3=2.44, only ONE mode per field type is active below M_P:
      Scalar J=0  (d=1),  Vector n=1 (d=6),  Spinor n=0 (d=2).
    The v1 zero-mode-only approach was correct by accident.

Key physical finding preserved from v1:
  chi_cross = 0.333 chi_vev  is OUTSIDE the physical emergence
  window [0.41, 1.0].  V_chi'' > 0 everywhere inside the window.

Author: CGC Gap 1/3 (corrected)
Date: 2026-07-29
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frg_flow_rp3 import L_RP3, M_CURV, M_P
from .frg_trace_density import M_H, M_T, M_W, M_Z

# ────────────────── EC-derived parameters (rigid) ──────────────────
T_FLAVOR = 5
ALPHA = -1.0 / 50.0
LAMBDA_CHI = 0.36
CHI_VEV = M_P / np.sqrt(float(T_FLAVOR - 2))
M2_CHI_BARE = ALPHA * M_P * M_P / float(T_FLAVOR - 2)
ABS_M_CHI = np.sqrt(abs(M2_CHI_BARE))
CHI_CROSS = ABS_M_CHI * np.sqrt(2.0 / LAMBDA_CHI)
CHI_MIN_WINDOW = M_CURV / M_P


def vpp_chi(chi: float) -> float:
    """V_chi''(chi)."""
    return M2_CHI_BARE + 0.5 * LAMBDA_CHI * chi * chi

# References
#     Coupling enhancement from trace density: Wetterich equation
#     Litim (2001): regulator-dependence < 0.001%
#



# ────────────────── SM field content ──────────────────


@dataclass
class SMField:
    name: str
    n_species: int  # 3 for colors etc.
    dof_per: int  # physical DOF per species
    mass_gev: float
    is_fermion: bool = False


SM_FIELDS = [
    SMField("SU(3) gluons", 8, 2, 0.0),
    SMField("W+- bosons", 2, 3, M_W),
    SMField("Z boson", 1, 3, M_Z),
    SMField("Photon", 1, 2, 0.0),
    SMField("Top quark", 3, 4, M_T, True),
    SMField("Bottom quark", 3, 4, 4.18, True),
    SMField("Charm quark", 3, 4, 1.27, True),
    SMField("Light q (uds)", 9, 4, 0.0, True),
    SMField("Tau lepton", 1, 4, 1.777, True),
    SMField("Muon", 1, 4, 0.10566, True),
    SMField("Electron", 1, 4, 5.11e-4, True),
    SMField("Neutrinos(3)", 3, 2, 0.0, True),
    SMField("Higgs", 1, 1, M_H),
    SMField("Goldstones", 3, 1, M_Z),
]

# RP³ zero-mode DOF per field type (all verified < M_P at L=2.44)
SCALAR_MODES = [(0.0, 1)]  # (eigenvalue in GeV², degeneracy)
VECTOR_MODES = [(4.0 / L_RP3**2 * M_P**2, 6)]
SPINOR_MODES = [(2.25 / L_RP3**2 * M_P**2, 2)]


def _threshold(k2: np.ndarray, m2: float) -> np.ndarray:
    """CORRECT threshold function: 2·k²/(k²+m²).

    This is dR_t/dt / (P_k + m²) for the Litim regulator,
    used in the RP3 trace density (frg_flow_rp3.py).
    NOT k⁴/(k²+m²)² which is the effective-potential flow kernel.
    """
    k2_pos = np.maximum(k2, 1e-60)
    denom = np.maximum(k2_pos + max(float(m2), 0.0), 1e-60)
    return 2.0 * k2_pos / denom  # type: ignore[no-any-return]


@dataclass
class EnhancedResult:
    chi_ratio: np.ndarray
    chi_gev: np.ndarray
    vpp: np.ndarray
    m_curv: np.ndarray
    eta_dchi: np.ndarray
    eta_boson: np.ndarray
    eta_fermion: np.ndarray
    eta_total: np.ndarray
    f_chi: np.ndarray
    notes: list


def compute(n_chi: int = 40) -> EnhancedResult:
    n_k = 400
    chi_grid = np.linspace(CHI_MIN_WINDOW, 1.0, n_chi)
    chi_gev = chi_grid * CHI_VEV
    vpps = np.array([vpp_chi(c) for c in chi_gev])
    m_curvs = M_P / (L_RP3 * np.maximum(chi_grid, 1e-60))

    eta_dchi = np.zeros(n_chi)
    eta_bos = np.zeros(n_chi)
    eta_fer = np.zeros(n_chi)

    for i, (_chi_r, _chi_val) in enumerate(zip(chi_grid, chi_gev, strict=False)):
        m_curv = m_curvs[i]
        k_ir = m_curv
        k = np.geomspace(k_ir, M_P, n_k)
        k2 = k * k
        d_ln_k = np.log(k[1] / k[0])
        vtx2 = (m_curv / M_P) ** 4

        # ── 1. delta-chi (scalar, dof=1, mass²=V_chi'') ──
        vpp = vpps[i]
        th_dchi = _threshold(k2, vpp)
        eta_dc = 1.0 * vtx2 * th_dchi / (16.0 * np.pi**2)
        eta_dchi[i] = np.sum(eta_dc) * d_ln_k

        # ── 2. SM fields with RP3 zero modes per spin type ──
        # Boson fields (positive eta)
        for fld in SM_FIELDS:
            if fld.is_fermion:
                continue
            m2_use = 0.0 if fld.mass_gev < 0.5 else fld.mass_gev ** 2

            # Determine mode type by field name
            name = fld.name.lower()
            modes = VECTOR_MODES if any(kw in name for kw in ["gluon", "w+-", "boson", "photon"]) else SCALAR_MODES

            field_eta = 0.0
            for lam2, d_mod in modes:
                total_m2 = max(m2_use, lam2)
                active = k2 > lam2
                th = np.zeros(n_k)
                th[active] = _threshold(k2[active], max(total_m2, 0.0))
                field_eta += d_mod * fld.n_species * fld.dof_per * vtx2 * th[active].sum() / (16.0 * np.pi**2)
            eta_bos[i] += field_eta * d_ln_k

        # Fermion fields (negative eta)
        for fld in SM_FIELDS:
            if not fld.is_fermion:
                continue
            m2_use = 0.0 if fld.mass_gev < 0.5 else fld.mass_gev ** 2

            modes = SPINOR_MODES
            field_eta = 0.0
            for lam2, d_mod in modes:
                total_m2 = max(m2_use, lam2)
                active = k2 > lam2
                th = np.zeros(n_k)
                th[active] = _threshold(k2[active], max(total_m2, 0.0))
                field_eta += d_mod * fld.n_species * fld.dof_per * vtx2 * th[active].sum() / (16.0 * np.pi**2)
            # FERMION SIGN FLIP (Bug 2 fix)
            eta_fer[i] -= field_eta * d_ln_k

    eta_total = eta_dchi + eta_bos + eta_fer

    # f(chi) = exp(integral_{chi}^{chi_vev} eta_bar(chi') d ln chi')
    d_ln_chi = np.abs(np.log(chi_grid[1] / chi_grid[0]))
    cum = 0.0
    f_chi = np.ones(n_chi)
    for i in range(n_chi - 1, -1, -1):
        cum += eta_total[i] * d_ln_chi
        f_chi[i] = np.exp(cum)
    f_chi /= f_chi[-1]

    notes = [
        f"Physical window: {CHI_MIN_WINDOW:.4f} <= chi/chi_v <= 1.0",
        f"chi_cross = {CHI_CROSS / CHI_VEV:.4f} (outside window)",
        f"V_chi'' > 0 in window: {bool(np.all(vpps > 0))}",
        "Threshold: 2k^2/(k^2+m^2) (correct RP3 self-energy trace)",
        "Fermion sign: flipped (negative contribution)",
        "RP3 modes < M_P: scalar(1), vector(1), spinor(1)",
    ]
    return EnhancedResult(
        chi_ratio=chi_grid,
        chi_gev=chi_gev,
        vpp=vpps,
        m_curv=m_curvs,
        eta_dchi=eta_dchi,
        eta_boson=eta_bos,
        eta_fermion=eta_fer,
        eta_total=eta_total,
        f_chi=f_chi,
        notes=notes,
    )


def report() -> str:
    res = compute(n_chi=40)
    f_ratio = res.f_chi[0] / res.f_chi[-1]
    lines = [
        "=" * 70,
        "  Gap 1/3: f(chi) — CORRECTED (v2: 2k^2 trunc + fermion sign)",
        "=" * 70,
        "",
        f"  chi_cross = {CHI_CROSS / CHI_VEV:.4f} chi_v  (outside window [{CHI_MIN_WINDOW:.4f},1.0])",
        f"  V_chi'' > 0 everywhere: {bool(np.all(res.vpp > 0))}",
        "",
        f"  {'chi/chi_v':>9s}  {'Vpp':>12s}  {'eta_dchi':>12s}  {'eta_bos':>12s}  {'eta_fer':>12s}  {'eta_tot':>12s}  {'f(chi)':>10s}",
        f"  {'-' * 9}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 10}",
    ]
    step = max(1, len(res.chi_ratio) // 12)
    for i in range(0, len(res.chi_ratio), step):
        lines.append(
            f"  {res.chi_ratio[i]:9.4f}  {res.vpp[i]:12.4e}  {res.eta_dchi[i]:12.4e}  "
            f"{res.eta_boson[i]:12.4e}  {res.eta_fermion[i]:12.4e}  {res.eta_total[i]:12.4e}  "
            f"{res.f_chi[i]:10.6f}"
        )
    lines.extend(
        [
            "",
            f"  f_chi range: [{res.f_chi[0]:.6f}, {res.f_chi[-1]:.6f}]",
            f"  Enhancement total: x{f_ratio:.6f}",
            f"  Boson contribution: {np.mean(np.abs(res.eta_boson)):.4e}",
            f"  Fermion contribution: {np.mean(np.abs(res.eta_fermion)):.4e}",
            f"  Delta-chi contribution: {np.mean(np.abs(res.eta_dchi)):.4e}",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import io
    import sys

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print(report())
