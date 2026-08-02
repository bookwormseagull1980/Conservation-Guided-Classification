# Extending CGC — Adding a New Operator Channel

This guide explains how to add a new operator channel to CGC. The four existing channels (`tmunu`, `f2`, `fermion`, `higgs`) serve as reusable templates.

## Overview

Adding a new operator involves three steps:
1. **Define the `OperatorSpec`** — name, operator type, field content
2. **Configure diagram generation** — which vertices and propagators to include
3. **Add to the command-line registry** — make it available via `cgc-run`

## Step 1: Define the Operator Specification

Create a new file in `cgc/channels/`, e.g., `cgc/channels/my_operator.py`:

```python
"""My Operator — description of what this operator represents."""

from cgc.engine.diagram_generator import OperatorSpec, OperatorType

# Define the operator specification
MY_OPERATOR = OperatorSpec(
    name="My Operator (description)",
    op_type=OperatorType.SCALAR,  # or CONSERVED_CURRENT, GAUGE_FIELD_STRENGTH, FERMION_BILINEAR
    field_content=[
        # List of fields participating in this operator's vertices.
        # Format: (field_type, multiplicity, spin)
        # field_type from: "gluon", "higgs", "fermion_up", "fermion_down", etc.
    ],
)

# Optional: pre-defined benchmark for validation
MY_OPERATOR_BENCHMARK = {
    "expected_protected": False,   # True if conservation law protects this operator
    "expected_pi0_sign": "positive",  # "positive", "negative", or "zero"
    "note": "Known from ...",
}
```

### Operator Types

| Type | Description | When to use |
|:---|:---|:---|
| `CONSERVED_CURRENT` | Conserved Noether current (e.g., Tμν, Jμ) | Operator satisfies ∂μJμ = 0 |
| `GAUGE_FIELD_STRENGTH` | Gauge-invariant field strength (e.g., F², FF̃) | Operator is BRST-invariant |
| `FERMION_BILINEAR` | Fermion bilinear (e.g., ψ̄ψ, ψ̄γ₅ψ, ψ̄γμψ) | Two-fermion operator |
| `SCALAR` | Scalar n-point function (e.g., φ⁴, φ²) | Bosonic self-interaction |

## Step 2: Configure Diagram Generation

### Field Content

The `field_content` list determines which one-loop diagrams are generated. Each entry specifies a field type and its multiplicity (number of flavors/colors) and spin:

```python
field_content=[
    # (field_type, multiplicity, spin)
    ("gluon", 8, 1),       # 8 gluons, spin-1
    ("higgs", 1, 0),        # 1 Higgs, spin-0
    ("fermion_up", 3, 0.5), # 3 colors of up-type, spin-1/2
]
```

### Choosing Field Content

- **For gauge interactions**: Include the gauge boson and all charged matter fields
- **For Higgs interactions**: Include the Higgs and all fields it couples to (top quark, W/Z, self)
- **For fermion operators**: Include the fermion species and any bosons that mediate the interaction

### Diagram Generation Options

The `OperatorSpec` supports additional configuration:

```python
OperatorSpec(
    ...
    is_1pi=True,           # Only 1PI diagrams (default)
    is_connected=True,      # Only connected diagrams (default)
    max_loops=1,            # One-loop by default
)
```

## Step 3: Register in the CLI

Add your operator to the `OPERATORS` dictionary in `cgc/benchmarks/run.py`:

```python
OPERATORS["my_op"] = {
    "name": "My Operator",
    "module": "cgc.channels.my_operator",
    "class": "MyOperatorClass",  # or use a factory function
    "description": "Brief description for --list-operators output",
    "is_conserved": False,
    "expected_protection": None,
}
```

If your operator is defined as a module-level constant (like `MY_OPERATOR` above), you can create a simple factory:

```python
# In cgc/channels/my_operator.py
def create() -> OperatorSpec:
    return MY_OPERATOR
```

Or wrap it in a class:

```python
class MyOperator(OperatorSpec):
    def __init__(self):
        super().__init__(...)
```

The `cgc/benchmarks/run.py` script uses `getattr(module, class_name)()` to instantiate.

## Validation

After adding your operator, verify it works:

```bash
cgc-run --list-operators          # Your operator should appear
cgc-run --operator my_op          # Run classification
```

Add a unit test in `cgc/tests/`:

```python
def test_my_operator_classification():
    from cgc.channels.my_operator import MY_OPERATOR
    from cgc import CGCPipeline

    pipeline = CGCPipeline()
    report = pipeline.run(MY_OPERATOR)
    assert report.conservation_report is not None
    assert report.diagram_set.total_count > 0
```

## Templates

Study the four existing channels for reference:

| Channel | File | Model |
|:---|:---|:---|
| Tμν spin-2 | `cgc/channels/tmunu_spin2.py` | Conserved current with Ward identity protection |
| F² gauge | `cgc/channels/gauge_field.py` | Gauge field strength with BRST protection |
| Fermion bilinears | `cgc/channels/fermion_bilinears.py` | Unprotected fermion operators |
| Higgs quartic | `cgc/channels/higgs_quartic.py` | Unprotected scalar operators |

The **protected** channels (tmunu, f2) are good models for conserved operators. The **unprotected** channels (fermion, higgs) are good models for operators without symmetry protection.

## Advanced: Custom Diagram Backend

The diagram generator supports two backends:
- `PYTHON_NATIVE` — Pure Python, self-contained
- `FEYNARTS` — Requires Mathematica + FeynArts installation

To add a custom backend, implement the `DiagramGenerator` interface in `cgc/engine/diagram_generator.py` and register it in the `GeneratorBackend` enum.
