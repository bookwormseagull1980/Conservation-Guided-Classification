# Conservation-Guided Classification (CGC)

**Classify quantum field theory operators by conservation-law protection status, without requiring coupling strength values.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

CGC is the computational implementation of the classification framework developed in:

> **Jinku Guo**, *"Coarse-Graining and the Classification of Long-Range Correlations in Quantum Field Theory,"* submitted to JHEP (2026).

## Quick Start

```bash
pip install -e .
cgc-run --operator tmunu
```

This classifies the energy-momentum tensor (Tμν) and prints:
- Whether it is **protected** by a conservation theorem (Ward identity)
- Whether its injection term **survives** at zero momentum transfer
- Diagram topology breakdown (single-bubble, ladder)
- Emergence interpretation (driving vs spectator)

```bash
# All four channels
cgc-run --operator f2       # Gauge field strength F^2
cgc-run --operator fermion  # Fermion bilinears
cgc-run --operator higgs    # Higgs quartic coupling

# Export full report
cgc-run --operator tmunu --output result.json
```

## What CGC Solves

In functional renormalization group (FRG) calculations for emergent gravity, the trace density β(σ) determines the IR behavior of spacetime. Each operator species contributes to β via the Wetterich equation, but **not all contributions are equal**.

CGC answers two questions:
1. **Classification**: Given an operator's field content and conservation properties, does its injection term survive at zero momentum transfer?
2. **Resummation**: Given the surviving diagrams, what is the resummed contribution to the trace density?

The key insight: conservation laws (Ward identities, BRST symmetry) **independently guarantee** that certain operator classes have non-vanishing matrix elements at q=0. This classification is **rigid** — it depends on symmetry, not on coupling magnitudes.

## Supported Operators

| Channel | Operator | Status | Protection Basis |
|:---|:---|:---|:---|
| `tmunu` | Tμν spin-2 (TT projection) | Protected | Ward identity |
| `f2` | Gauge field strength F² | Protected | BRST symmetry |
| `fermion` | Fermion bilinears (ψ̄ψ, ψ̄γ₅ψ) | Unprotected | None |
| `higgs` | Higgs quartic φ⁴ | Unprotected | None |

## Validation Framework

CGC is verified by a **5-layer validation network** (126 unit tests + benchmark layers):

| Layer | Description | Checks | Status |
|:---|:---|:---|:---|
| L0 | Unit tests | 126 | ✅ |
| L1 | Internal self-consistency | 4 | ✅ |
| L2 | Cross-validation | 2 | ✅ |
| L4 | Known-model benchmarks | 5 | ✅ |
| STABILITY | Numerical stability | 5 dimensions | ✅ |

Run all checks:
```bash
cgc-benchmark
```

## Two Computation Components

The repository contains two distinct Π₀ computations, clearly separated:

1. **Flat-space single-bubble table** (`cgc/engine/pi0_flat_continuum.py`, v3) —
   the values quoted in the paper (sec05, eq:pi0def): Gaussian cutoff Λ²=1,
   **bare normalisation** (no coupling constants), massless limit,
   spin-statistics-weighted mode counting. No external parameters.
   F² = −0.3546, G² = +0.1013, J^μ = −0.2026, Tμν = 0 in flat spacetime.
2. **RP³ FRG cross-validation** (`cgc/rp3_engine/` — `frg_flow_rp3.py`,
   `frg_trace_density.py`, `self_consistent_dyson.py`, etc.) —
   the FRG engine on the compact internal space, kept as a SEPARATE
   subpackage (Paper 3 lineage), not part of the flat-space CGC core.
   Its `pi0_bare_ir` values are geometry-dependent and agree with the
   classification only at the level of signs.

Only the signs of Π₀ are classification-relevant; magnitudes are
scheme-dependent (as stated in the paper).

### Isolation guarantee

The flat-space CGC core (`cgc/engine/`) is fully independent of the
RP³ subpackage: the paper's Π₀ values are fixed by SM field content,
spin-statistics and charges (bare normalisation, no couplings, no
gravity factors).  `cgc/engine/` imports nothing from `cgc/rp3_engine/`;
the RP³ engine (including its convention-specific coefficient
`c_T = 3/4` in `gravity_feedback.py`) is used only for sign-level
cross-validation and never feeds the core classification.

## CLI Reference

### `cgc-run` — Classify an operator

```
Usage: cgc-run [--operator {tmunu|f2|fermion|higgs}] [--output FILE] [--list-operators]
```

### `cgc-verify` — Verify against reference benchmarks

```
Usage: cgc-verify [--verbose] [--tolerance 1e-8]
```

Compares current output against `reference_output.json` across L1, L2, and L4 layers.

### `cgc-benchmark` — Run all validation layers

```
Usage: cgc-benchmark [--layer {L0|L1|L2|L4|STABILITY|all}] [--verbose]
```

## Installation

### From Source

```bash
git clone https://github.com/bookwormseagull1980/Conservation-Guided-Classification.git
cd Conservation-Guided-Classification
pip install -e ".[dev]"
```

### Requirements

- Python ≥ 3.10
- numpy ≥ 1.24
- scipy ≥ 1.10

Development dependencies: pytest, hypothesis, mypy, ruff.

## Project Structure

```
cgc/
├── engine/          # Core classification engine (FLAT SPACETIME)
│   ├── diagram_generator.py    # Feynman diagram generation
│   ├── momentum_classifier.py  # Momentum transfer (q=0 vs q≠0)
│   ├── topology_classifier.py  # Topology (bubble vs ladder)
│   ├── conservation_checker.py # Conservation law assessment
│   ├── resummation.py          # Ladder resummation
│   ├── pi0_flat_continuum.py   # Flat-space single-bubble Π₀ (paper values, v3)
│   └── pipeline.py             # Full classification pipeline
├── rp3_engine/      # RP³ FRG cross-validation (SEPARATE subpackage)
│   ├── frg_flow_rp3.py         # RP³ FRG flow solver
│   ├── frg_trace_density.py    # RP³ trace density
│   └── ...                     # Paper 3 lineage, not part of CGC core
├── channels/        # Operator-specific configurations
│   ├── tmunu_spin2.py
│   ├── gauge_field.py
│   ├── fermion_bilinears.py
│   └── higgs_quartic.py
├── benchmarks/      # Validation infrastructure
│   ├── verify.py                # Reference comparison
│   ├── benchmark_runner.py      # Full benchmark suite
│   ├── model_benchmarks.py      # Known-model checks
│   └── numerical_stability.py   # Stability analysis
├── interface/       # CG-Framework bridge
│   ├── schema.py    # JSON schema (v1.1.0)
│   └── bridge.py    # CGC↔FRG data exchange
└── tests/           # Unit tests (126 tests)
```

## Citing CGC

If you use CGC in your research, please cite:

```bibtex
@article{Guo:2026cgc,
  author  = {Jinku Guo},
  title   = {{Coarse-Graining and the Classification of
              Long-Range Correlations in Quantum Field Theory}},
  note    = {submitted to JHEP},
  year    = {2026},
}
```

The foundational framework on which CGC builds:

```bibtex
@article{Guo:2026spectral,
  author  = {J.~K. Guo},
  title   = {{A spectral criterion for emergent gravity}},
  year    = {2026},
  eprint  = {2607.21621},
  archivePrefix = {arXiv},
}
```

## License

MIT — see [LICENSE](./LICENSE).
