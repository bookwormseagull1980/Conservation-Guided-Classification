# Known Limitations

This document honestly documents the known limitations of the CGC framework as of v1.1.0. These are not bugs — they are inherent to the current scope of the method and the assumptions on which it rests.

## 1. Ladder Approximation

CGC's resummation engine operates in the **ladder approximation**: only diagrams with the ladder topology are resummed. Cross-ladder, vertex-correction, and overlapping-loop diagrams are classified but **not** included in the resummed contribution.

**What this means**: The resummation result is exact within the ladder subset but incomplete with respect to the full diagram series. The ladder approximation is known to be exact for certain operator classes (e.g., large-N limits, certain kinematic regimes), but its completeness for general operators in emergent gravity has not been proven.

**Status**: The completeness theorem for non-ladder diagrams is an **open problem**. See `cgc/engine/resummation.py` for the current implementation.

## 2. Non-Ladder Completeness Theorem

CGC classifies all generated diagrams by topology (single-bubble, ladder, cross-ladder, vertex correction, overlapping), but the **resummation** step only handles ladder diagrams. Proving that non-ladder contributions are sub-leading (or vanishing) in the specific FRG context of emergent gravity on RP³ requires a dedicated analytic study that has not been completed.

**What this means**: The classification is complete (all diagrams are categorized), but the resummed result should be interpreted as a **lower bound** on the true contribution until the completeness theorem is established.

## 3. FRG Scheme Dependence

The Π₀(0) values used by CGC for cross-validation are computed using a **Litim regulator** and the Wetterich equation in the local potential approximation (LPA). Different regulator choices can shift these values. (Note the distinction: the flat-space single-bubble values quoted in the paper, sec05 eq:pi0def, use the **Gaussian cutoff** with bare normalisation and are computed by `pi0_flat_continuum.py`; the Litim values belong to the RP³ FRG cross-validation path, reported separately.)

**Mitigation**: Numerical stability analysis (see `cgc/benchmarks/numerical_stability.py`) confirms that the Litim vs exponential regulator difference is **< 0.001%**, far below the physical precision threshold. Additionally, the key classification results (protected vs unprotected, nonzero vs vanishing injection) are **topological** and do not depend on the regulator choice.

**What this means**: The regulator dependence is a quantitative uncertainty, not a qualitative one. The classification verdicts are regulator-independent.

## 4. CGC vs FRG — Division of Labor

CGC and FRG serve **complementary** roles:

| | CGC | FRG |
|:---|:---|:---|
| **Method** | Analytic classification | Numerical integration |
| **Input** | Operator field content + conservation law | Action + regulator |
| **Output** | Protected/unprotected verdict, diagram counts | β-function, fixed-point structure |
| **Uncertainty** | Completeness of diagram series | Regulator dependence, truncation error |

CGC provides the **analytic classification** — it tells you *which* contributions matter and *why* (symmetry protection). FRG provides the **numerical verification** — it computes *how much* each contribution affects the flow.

**What this means**: CGC is not a replacement for FRG. It is a complementary tool that provides understanding and verification that FRG alone cannot offer.

## 5. Field Content Configuration

The current implementation supports the Standard Model field content (gluons, W/Z, fermions, Higgs) as the input to diagram generation. Adding new physics (e.g., dark sector fields, heavy right-handed neutrinos, axions) requires:
1. Defining new field species in `cgc/engine/diagram_generator.py`
2. Specifying their interaction vertices in the model file
3. Validating against known benchmarks

See `docs/extending.md` for instructions.

## 6. One-Loop Scope

The current diagram generator produces **one-loop** diagrams (bubbles, triangles, boxes). While momentum classification and topology classification extend to higher orders in principle, the diagram generation is limited to one-loop order. Multi-loop extensions are conceptually straightforward but not implemented.

**Status**: This is sufficient for the current application domain (FRG β-functions in the LPA approximation, where one-loop diagrams dominate). The framework is structured to accommodate multi-loop extensions.

## 7. Operator Channels

Four operator channels are currently implemented:
- Tμν (energy-momentum tensor, spin-2 TT projection)
- F² (gauge field strength)
- Fermion bilinears
- Higgs quartic

Additional channels (e.g., Chern-Simons terms, topological operators, mixed fermion-boson operators) can be added following the recipe in `docs/extending.md`.
