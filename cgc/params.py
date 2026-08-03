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
  G1_MG  : g_1(M_G) = 0.6083 (KV + GUT normalization + SM RGE back-fit)

FLAT-SPACE SINGLE-BUBBLE TABLE (v3, fixed 2026-08-01):
  The 5-channel flat-space Pi0 table (pi0_flat_continuum.py v3) uses
  BARE normalisation: operator coefficient = 1, NO coupling^2 factors,
  Gaussian cutoff Lambda^2=1, massless limit.  It does NOT import any
  coupling or scale from this file.  See paper3-1 sec05 (eq:pi0def).

RP3 FRG CROSS-VALIDATION COUPLINGS:
  The coupling constants below (G3_MG, G2_MG, G1_MG, G3_SQ, G2_SQ)
  serve the RP3 FRG cross-validation modules (frg_flow_rp3.py,
  frg_trace_density.py, self_consistent_dyson.py) — a separate
  component whose SIGNS agree with the classification verdicts
  (magnitudes are scheme-dependent; reported separately).

All energies in GeV.
"""

# ── Fundamental scales ──────────────────────────────────────────────
M_P = 2.4353e18        # reduced Planck mass [GeV]  (CODATA 2022)
M_G = M_P / 1.438      # emergence scale [GeV] = 1.6935e18 (L_G/L_C closure)
L_RP3 = 2.44           # RP3 radius at σ_G (dimensionless, Planck units)
M_CURV = M_P / L_RP3   # curvature mass scale [GeV] ~ 1e18

# ── Gauge couplings at M_G ──────────────────────────────────────────
G3_MG = 0.496          # SU(3): Cartan generator on EC connection (−0.40% vs SM)
G2_MG = 0.516          # SU(2)_L: KV zero modes on RP3 (+1.6% vs SM)
G1_MG = 0.6083          # U(1)_Y at M_G: KV + GUT normalization + κ=√(5/6) SM RGE back-fit

# ── Channel coupling² (fixed convention) ────────────────────────────
G3_SQ = G3_MG**2       # F², G² channel coupling² = 0.246
G2_SQ = G2_MG**2       # Jᵘ channel coupling² = 0.266

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


def summary() -> str:
    return (
        "CGC canonical parameters:\n"
        f"  M_P    = {M_P:.6e} GeV\n"
        f"  M_G    = {M_G:.6e} GeV\n"
        f"  g3(MG) = {G3_MG}  (g3² = {G3_SQ:.6f})\n"
        f"  g2(MG) = {G2_MG}  (g2² = {G2_SQ:.6f})\n"
        f"  g1(MG) = {G1_MG}\n"
    )


if __name__ == "__main__":
    print(summary())
