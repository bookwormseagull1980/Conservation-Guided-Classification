"""Pi0(0) — flat-space continuum single-bubble integrals (CGC Phase 4, v3).

=====================================================================
DEFINITION (2026-08-01, hardened for publication)
=====================================================================
The framework's leading-order single-bubble kernel is DEFINED as

    Pi0 = sum_fields  s_f * n_f * K_O * I_G(m_f^2)

with the conventions of paper3-1 sec05:

  1. Continuum limit, flat spacetime, Euclidean, Gaussian cutoff
     e^{-p^2/Lambda^2} with Lambda^2 = 1 in dimensionless units.
  2. Bare operator normalisation: the composite operator carries
     coefficient 1 (no alpha_s/pi, no 1/g^2, no coupling^2 factors).
  3. s_f = +1 for boson loops, -1 for fermion loops (spin-statistics).
     The SIGN of Pi0 is the classification-relevant quantity.
  4. n_f = number of field modes of species f:
       gluons       16  (8 colours x 2 helicities)
       quarks       12 per flavour (3 colours x 4 Dirac components)
       charged lept  4 per flavour
       neutrinos     2 per flavour
  5. K_O = operator kernel: 1 for Lorentz-scalar operators (F^2, G^2);
     Q_f^2 for the conserved vector current J^mu = sum_f Q_f psibar
     gamma^mu psi_f (the EM current), whose bubble carries the charge
     square of each charged fermion species.
  6. All Standard Model masses satisfy m_f << Lambda, so fields are
     taken massless and

         I_G(0) = 1/(16 pi^2)

     (the Gaussian scalar bubble; masses enter only through
      I_G(m^2) = (1/16 pi^2)[1 - m^2 e^{m^2} E_1(m^2)], E_1 = Gamma(0,x),
      and the massless limit is the one quoted in the paper).
  7. T_munu (spin-2 and spin-0): Pi0 = 0 exactly in flat spacetime by
     the Ward identity d^mu T_munu = 0 (a conserved current's
     zero-momentum two-point function vanishes; a non-zero value
     requires geometry and is not computed in this framework).

Pi0 is a CLASSIFICATION DIAGNOSTIC (defined quantity), not a physical
cross-section: only its sign is scheme-independent, and the magnitude
is scheme-dependent (as stated in the paper).

Values produced by THIS module (v3, massless limit):

    F2 = -0.3546   G2 = +0.1013   Ju = -0.2026
    Tmunu spin-2 = 0   Tmunu spin-0 = 0   (Ward identity)

v1/v2 magnitudes (LiTim+coupling^2: -0.157/+0.117/-0.440;
Gaussian+bare+masses-in-GeV: -0.1545/+0.1013/-0.4659) are superseded;
the paper quotes the v3 values.  The sign structure is unchanged and
matches the classification table.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
import numpy as np
from scipy.special import exp1

# Allow direct script execution: add repo root to sys.path
_CGC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _CGC_ROOT not in sys.path:
    sys.path.insert(0, _CGC_ROOT)


def I_gauss(m2: float) -> float:
    r"""Gaussian-cutoff scalar bubble, dimensionless mass squared.

    I_G(m^2) = (1/16 pi^2) [1 - m^2 e^{m^2} E_1(m^2)]

    Massless limit (m^2 -> 0): I_G -> 1/16 pi^2.
    """
    if m2 < 1e-12:
        return 1.0 / (16.0 * np.pi**2)
    if m2 > 50.0:  # asymptotic e^x E_1(x) ~ 1/x (1 - 1/x + 2!/x^2 - ...)
        x = m2
        xe = 1.0 - 1.0 / x + 2.0 / x**2 - 6.0 / x**3
    else:
        xe = m2 * np.exp(m2) * exp1(m2)
    return (1.0 - xe) / (16.0 * np.pi**2)


@dataclass
class Field:
    name: str
    is_fermion: bool
    n_dof: int
    kernel: float = 1.0   # K_O: 1 (scalar operators) or Q_f^2 (J^mu)
    mass: float = 0.0     # GeV; m_f << Lambda -> massless limit


def _f2_fields() -> list:
    """F²: SU(3) field strength — gluons + quarks, K=1."""
    return [
        Field("SU(3) gluons",  False, 16),
        Field("u,d,s quarks",  True,  36),
        Field("c quark",       True,  12),
        Field("b quark",       True,  12),
        Field("t quark",       True,  12),
    ]


def _g2_fields() -> list:
    """G²: scalar glueball — pure gluon, K=1."""
    return [
        Field("SU(3) gluons",  False, 16),
    ]


def _ju_fields() -> list:
    """J^mu: EM current — charged fermions only, kernel Q_f^2.

    J^mu = sum_f Q_f psibar_f gamma^mu psi_f couples exclusively to
    charged fermions (neutrinos carry Q=0 and are excluded).  The
    bubble carries the charge square Q_f^2 of each species.

    NOTE (2026-08-03): the kernel values below ARE the charge squares
    Q_f^2 (e.g. u: (2/3)^2 = 4/9).  The variable is named Q_SQ to
    remove the earlier ambiguity where Q_f vs Q_f^2 was unclear.
    """
    Q_SQ = {"u": 4.0 / 9.0, "d": 1.0 / 9.0, "s": 1.0 / 9.0,
            "c": 4.0 / 9.0, "b": 1.0 / 9.0, "t": 4.0 / 9.0,
            "e": 1.0, "mu": 1.0, "tau": 1.0}
    dof = {"u": 12, "d": 12, "s": 12, "c": 12, "b": 12, "t": 12,
           "e": 4, "mu": 4, "tau": 4}
    return [
        Field(f"{name} (Q^2={Q_SQ[name]:.4g})", True, dof[name],
              kernel=Q_SQ[name])
        for name in ("u", "d", "s", "c", "b", "t", "e", "mu", "tau")
    ]


@dataclass
class Pi0Result:
    channel: str
    tag: str
    pi0: float
    sign: int
    breakdown: list = field(default_factory=list)


def compute(channel: str, tag: str, fields: list) -> Pi0Result:
    """Pi0 = sum_fields s_f * n_f * K_O * I_G(m^2), massless limit."""
    total = 0.0
    contribs = []
    for f in fields:
        s = -1.0 if f.is_fermion else +1.0
        I = I_gauss(f.mass * f.mass)
        c = s * f.n_dof * f.kernel * I
        total += c
        contribs.append(dict(name=f.name, dof=f.n_dof, ferm=f.is_fermion,
                             m=f.mass, K=f.kernel, I=I, c=c))
    return Pi0Result(channel=channel, tag=tag, pi0=total,
                     sign=1 if total > 1e-12 else (-1 if total < -1e-12 else 0),
                     breakdown=contribs)


def compute_all() -> list:
    """All five channels (Tmunu = 0 exactly by the Ward identity)."""
    return [
        compute("F2 (gauge field strength)", "F2", _f2_fields()),
        Pi0Result("Tmunu (spin-2, TT)", "Tmunu_S2", 0.0, 0,
                  [dict(name="Ward identity: Pi0 = 0 in flat spacetime",
                        dof=0, ferm=False, m=0.0, K=0.0, I=0.0, c=0.0)]),
        Pi0Result("Tmunu (spin-0, tr)", "Tmunu_S0", 0.0, 0,
                  [dict(name="Ward identity: Pi0 = 0 in flat spacetime",
                        dof=0, ferm=False, m=0.0, K=0.0, I=0.0, c=0.0)]),
        compute("Ju (EM current)", "Ju", _ju_fields()),
        compute("G2 (scalar glueball)", "G2", _g2_fields()),
    ]


def fmt(results: list) -> str:
    L = []
    L.append("=" * 78)
    L.append("  Pi0(0) — flat-space continuum single-bubble kernel (v3)")
    L.append("  Gaussian Lambda^2=1, bare normalisation, massless limit")
    L.append("  Pi0 = sum s_f n_f K_O I_G; sign = classification-relevant")
    L.append("=" * 78)
    L.append("")
    L.append(f"  {'Channel':<30} {'Pi0':>14} {'Sign':>6}")
    L.append(f"  {'-'*30} {'-'*14} {'-'*6}")
    for r in results:
        s = "+" if r.sign > 0 else ("-" if r.sign < 0 else "0")
        L.append(f"  {r.channel:<30} {r.pi0:>14.6e} {s:>6}")
    L.append("")
    for r in results:
        L.append(f"  --- {r.channel} ---")
        for c in r.breakdown:
            if c['dof'] == 0 and abs(c['c']) < 1e-12:
                L.append(f"  {c['name']}")
                continue
            fb = "F" if c['ferm'] else "B"
            L.append(f"  {c['name']:<28} {c['dof']:>4} {fb:>4} K={c['K']:>6.4g}"
                     f" {c['I']:>12.6e} {c['c']:>+14.6e}")
        L.append(f"  {'TOTAL':<28} {'':>4} {'':>4} {'':>12} {'':>12} {r.pi0:>+14.6e}")
        L.append("")
    L.append("  Sign check (classification-relevant):")
    expected = {"F2": "-", "Tmunu_S2": "0", "Tmunu_S0": "0", "Ju": "-", "G2": "+"}
    for r in results:
        ok = "+" if r.sign > 0 else ("-" if r.sign < 0 else "0")
        ex = expected[r.tag]
        match = "MATCH" if ok == ex else "MISMATCH"
        L.append(f"  {r.tag:<10} computed={ok} expected={ex}  [{match}]")
    return "\n".join(L)


if __name__ == "__main__":
    results = compute_all()
    print(fmt(results))
