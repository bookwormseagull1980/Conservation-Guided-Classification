# Archived RP³-path outputs (2026-08-01)

These JSON files are artifacts of the **RP³ FRG cross-validation path**
(Paper 3 lineage: `frg_flow_rp3.py`, `frg_trace_density.py`,
`self_consistent_dyson.py`), generated 2026-07-30 with the then-current
coupling² convention (g₃², g₂², 1/L⁴).

They are **archived, not deleted**, because:

1. Their `pi0_bare_ir` values (e.g. Tμν +0.0356, F² −0.1326) are the
   **geometry-dependent RP³ values**, distinct from the flat-space
   single-bubble values quoted in paper3-1 (F² −0.3546, G² +0.1013,
   J^μ −0.2026, Tμν = 0 in flat spacetime).
2. The RP³ cross-validation is reported separately; when it is updated
   to the current conventions, these files should be regenerated.

Consistency note (2026-08-01):
- The paper's flat-space Π₀ table comes from
  `cgc/engine/pi0_flat_continuum.py` (v3): bare normalisation, Gaussian
  cutoff Λ²=1, massless limit — no external parameters.
- The RP³ FRG machinery uses `cgc/params.py` couplings and agrees with
  the classification only at the level of signs (magnitudes are
  scheme-dependent, as stated in the paper).
