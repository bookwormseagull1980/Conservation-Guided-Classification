"""cgc.rp3_engine — RP³ internal-space FRG cross-validation (separate component).

2026-08-03: RP³ content separated from the flat-space CGC core.
The CGC classification core (cgc/engine/) computes in FLAT spacetime
and is fully independent of this subpackage:

  • cgc/engine/  (classification core, paper 3-1 values):
      - pi0_flat_continuum.py  — flat-space single-bubble Π₀ table
        (F² = −0.3546, J^μ = −0.2026, G² = +0.1013, Tμν = 0),
        bare normalisation, no coupling constants, no gravity factors.
      - conservation_checker, diagram_generator, momentum/topology
        classifiers, resummation, pipeline — conservation-law
        classification in flat spacetime.
      - NO imports from cgc.rp3_engine; no c_T, no gravity feedback,
        no RP³ spectrum enters the core.

  • cgc/rp3_engine/  (RP³ FRG cross-validation, Paper 3 lineage):
      - frg_flow_rp3, frg_trace_density, gravity_feedback (c_T = 3/4
        is a convention-specific effective coefficient inside this
        component), dyson_schwinger, chi_potential, crossed_ladder_f2.
      - Used only for sign-level cross-validation; magnitudes are
        scheme/convention-dependent and reported separately.

The two components share no physics inputs: the core's Π₀ values are
fixed by SM field content, spin-statistics and charges; the RP³ engine
has its own coupling set (G3_MG, G2_MG, G1_MG in cgc/params.py).
"""
