# Channel Coupling Conventions

## Conventions (2026-08-01, v3 — hardened for publication)

Single-bubble kernel **defined** as (paper3-1 sec05, Eq. (pi0def)):

    Pi0 = sum_fields  s_f * n_f * K_O * I_G(m_f^2)

- Gaussian cutoff e^{-p²/Λ²}, Λ² = 1 (dimensionless)
- **Bare operator normalisation**: operator coefficient = 1
  (no g², no 1/L⁴, no α_s/π)
- s_f = +1 boson, −1 fermion (spin statistics) — the sign is the
  classification-relevant quantity; the magnitude is scheme-dependent
- n_f = field modes: gluons 16 (8×2), quarks 12/flavour (3×4),
  charged leptons 4/flavour
- K_O = operator kernel: 1 for Lorentz-scalar operators (F², G²);
  Q_f² for J^μ (EM current, charged fermions only)
- Massless limit: all m_f ≪ Λ ⇒ I_G = 1/16π²
- Tμν (spin-2, spin-0): Pi0 = 0 in flat spacetime (Ward identity)

## Reference Pi0 Values (v3, massless limit)

| Channel | Π₀ | Sign |
|---------|-----|------|
| F² | −0.3546 | Negative |
| G² | +0.1013 | Positive |
| J^μ | −0.2026 | Negative |
| Tμν S2 | 0 (flat spacetime) | 0 |
| Tμν S0 | 0 (flat spacetime) | 0 |

Reproducible via `pi0_flat_continuum.py` (v3).  Paper tables
(sec03/sec05/sec06/combined_audit) quote these values.

## History

- 2026-08-01 v3: massless limit + explicit definition formula added to
  the paper; J^μ kernel Q_f² (trace claim removed — it was internally
  inconsistent: δ_{μν}tr[...] = −8 times the fermion (−1) is +, so the
  trace does not fix the negative sign; the sign comes from spin
  statistics).  Values: F² −0.3546, G² +0.1013, J^μ −0.2026.
- 2026-08-01 v2 (superseded): Gaussian + bare + masses in GeV
  (Λ = 1 GeV was arbitrary) — F² −0.1545, G² +0.1013, J^μ −0.4659.
- 2026-08-01 v1 (superseded): LiTim + coupling² (g₃², g₂², 1/L⁴) —
  docstring claimed "bare" but multiplied couplings.
- Pre-2026-08-01: magnitudes (−1.070, +0.0711, −0.138; Tμν S2
  +1.834×10⁻², S0 +4.918×10⁻⁴) had **no surviving computational
  source** and were removed.
