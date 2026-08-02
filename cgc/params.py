"""cgc/params.py — CGC Canonical Parameters (Single Source of Truth).

ALL CGC engine modules MUST import couplings/scales from here.
No hardcoded values anywhere else.  This file is the fix point for
"results keep changing" — one set of rigid values, imported everywhere.

Provenance (all from CG-Framework rigid derivation / PDG 2024):
  M_P    : [CODATA 2022] reduced Planck mass M_P = (8πG_N)^(-1/2)
  M_G    : emergence scale, M_G = M_P/1.438 (L_G/L_C closure)
  L_RP3  : RP3 radius at σ_G, MaxEnt + EWSB closure (A/B=1)
  G3_MG  : g_3(M_G) = 0.496  (Cartan generator on EC connection)
  G2_MG  : g_2(M_G) = 0.516  (KV zero modes on RP3)
  G1_MG  : g_1(M_G) = 0.666  (KV + GUT normalization)  [frg_trace_density]
  G1_MG_canon: 0.6083        (CG-Framework framework_params.py canonical;
                           frg_trace_density uses 0.666 — see note below)

NOTE on G1 discrepancy:
  CG-Framework cg_core/framework_params.py has g1_MG = 0.608255.
  CGC frg_trace_density.py historically used G1_MG = 0.666 (older value).
  This file keeps BOTH, marked clearly, so each module imports the
  value matching its derivation.

FLAT-SPACE SINGLE-BUBBLE TABLE (v3, fixed 2026-08-01):
  The 5-channel flat-space Pi0 table (pi0_flat_continuum.py v3) uses
  BARE normalisation: operator coefficient = 1, NO coupling^2 factors,
  Gaussian cutoff Lambda^2=1, massless limit.  It does NOT import any
  coupling or scale from this file.  See paper3-1 sec05 (eq:pi0def).

RP3 FRG CROSS-VALIDATION COUPLINGS (unchanged):
  The coupling constants below (G3_MG, G2_MG, G1_MG, G3_SQ, G2_SQ)
  serve the RP3 FRG cross-validation modules (frg_flow_rp3.py,
  frg_trace_density.py, self_consistent_dyson.py) — a separate
  component whose SIGNS agree with the classification verdicts
  (magnitudes are scheme-dependent; reported separately).

GRAV_SQ / channel_couplings() are the HISTORICAL v1 convention
(1/L^4 curvature vertex) — superseded for the flat-space table, where
T_munu Pi0 = 0 in flat spacetime by the Ward identity.  Retained only
for backward compatibility; no module imports them.

All energies in GeV.  Last updated: 2026-08-02 00:05.
"""

# ── Fundamental scales ──────────────────────────────────────────────
M_P = 2.4353e18        # reduced Planck mass [GeV]  (CODATA 2022)
M_G = M_P / 1.438      # emergence scale [GeV] = 1.6935e18 (L_G/L_C closure)
L_RP3 = 2.44           # RP3 radius at σ_G (dimensionless, Planck units)
M_CURV = M_P / L_RP3   # curvature mass scale [GeV] ~ 1e18

# ── Gauge couplings at M_G ──────────────────────────────────────────
G3_MG = 0.496          # SU(3): Cartan generator on EC connection (−0.40% vs SM)
G2_MG = 0.516          # SU(2)_L: KV zero modes on RP3 (+1.6% vs SM)
G1_MG = 0.666          # U(1)_Y: KV + GUT normalization (historical CGC value)
G1_MG_canon = 0.6083   # U(1)_Y: CG-Framework canonical (framework_params.py)

# ── Channel coupling² (fixed convention) ────────────────────────────
G3_SQ = G3_MG**2       # F², G² channel coupling² = 0.246
G2_SQ = G2_MG**2       # Jᵘ channel coupling² = 0.266
GRAV_SQ = 1.0 / L_RP3**4  # Tμν channel coupling² = 0.0282 (curvature vertex)

# ── SM masses [GeV] (PDG 2024) ──────────────────────────────────────
M_T = 172.76
M_B = 4.18
M_C = 1.27
M_TAU = 1.777
M_MU = 0.10566
M_E = 0.000511
M_H = 125.10
M_W = 80.377
M_Z = 91.1876

# Convenience
def channel_couplings() -> dict:
    """HISTORICAL v1 convention (superseded 2026-08-01).

    Coupling^2 values of the old flat-space Pi0 table.  The current
    flat-space computation (pi0_flat_continuum.py v3) is BARE and does
    not use these; the RP3 FRG cross-validation modules import G3_SQ /
    G2_SQ directly.  Retained only for backward compatibility.
    """
    return {
        "F2": G3_SQ,          # historical: gauge field strength
        "G2": G3_SQ,          # historical: glueball
        "Ju": G2_SQ,          # historical: fermion bilinear current
        "Tmunu_S2": GRAV_SQ,  # historical: 1/L^4 curvature vertex
        "Tmunu_S0": GRAV_SQ,  # historical: 1/L^4 curvature vertex
    }


def summary() -> str:
    return (
        "CGC canonical parameters:\n"
        f"  M_P    = {M_P:.6e} GeV\n"
        f"  M_G    = {M_G:.6e} GeV\n"
        f"  L_RP3  = {L_RP3}\n"
        f"  g3(MG) = {G3_MG}  (g3² = {G3_SQ:.6f})\n"
        f"  g2(MG) = {G2_MG}  (g2² = {G2_SQ:.6f})\n"
        f"  g1(MG) = {G1_MG}  (canon: {G1_MG_canon})\n"
        f"  1/L⁴    = {GRAV_SQ:.6f}  (Tμν curvature vertex)\n"
    )


if __name__ == "__main__":
    print(summary())
    print("Channel coupling²:", channel_couplings())
