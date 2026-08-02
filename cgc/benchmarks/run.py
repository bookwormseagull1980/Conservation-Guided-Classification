#!/usr/bin/env python3
"""cgc-run — Classify an operator's conservation status and emergence properties.

Usage:
    cgc-run --operator tmunu
    cgc-run --operator f2
    cgc-run --operator fermion
    cgc-run --operator higgs
    cgc-run --operator tmunu --output result.json
    cgc-run --list-operators

Supported operators (4 channels):
  tmunu    — Energy-momentum tensor (TT projection, spin-2)
  f2       — Gauge field strength F^2
  fermion  — Fermion bilinears
  higgs    — Higgs quartic coupling

Output: injection-term verdict, Pi0 sign and magnitude, emergence classification,
and key numerical values printed to terminal. Optional --output exports full report.
"""

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Ensure cgc package is importable from project root
_CGC_ROOT = Path(__file__).resolve().parents[2]
if str(_CGC_ROOT) not in sys.path:
    sys.path.insert(0, str(_CGC_ROOT))

# ── Operator registry ────────────────────────────────────────────────────

OPERATORS: dict[str, dict[str, str | bool | None]] = {
    "tmunu": {
        "name": "Tμν spin-2 (TT projection)",
        "module": "cgc.channels.tmunu_spin2",
        "class": "TMunuSpin2",
        "description": "Energy-momentum tensor, traceless-transverse projection",
        "is_conserved": True,
        "expected_protection": "WARD_IDENTITY",
    },
    "f2": {
        "name": "Gauge Field Strength F²",
        "module": "cgc.channels.gauge_field",
        "class": "GaugeFieldStrength",
        "description": "Gauge field strength squared, BRST-exact up to surface terms",
        "is_conserved": True,
        "expected_protection": "BRST_SYMMETRY",
    },
    "fermion": {
        "name": "Fermion Bilinears",
        "module": "cgc.channels.fermion_bilinears",
        "class": "FermionBilinears",
        "description": "ψ̄ψ, ψ̄γ₅ψ, and related scalar/pseudoscalar bilinears",
        "is_conserved": False,
        "expected_protection": None,
    },
    "higgs": {
        "name": "Higgs Quartic φ⁴",
        "module": "cgc.channels.higgs_quartic",
        "class": "HiggsQuartic",
        "description": "Scalar quartic self-coupling (Higgs portal)",
        "is_conserved": False,
        "expected_protection": None,
    },
}


def load_operator(key: str) -> Any:
    """Load an operator instance from the registry."""
    entry = OPERATORS[key]
    mod = __import__(entry["module"], fromlist=[entry["class"]])
    return getattr(mod, entry["class"])()


def run_classification(op_key: str) -> dict:
    """Run CGC classification pipeline and return structured results."""
    from cgc import CGCPipeline

    op = load_operator(op_key)
    pipeline = CGCPipeline()
    report = pipeline.run(op)

    v = report.conservation_report.verdict if report.conservation_report else None
    mc = report.momentum_classification
    tc = report.topology_classification

    result: dict[str, Any] = {
        "operator": op_key,
        "operator_name": op.name if hasattr(op, "name") else str(op),
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": report.pipeline_version,
        "classification": {
            "is_protected": v.is_protected if v else False,
            "matrix_element_nonzero": v.matrix_element_nonzero if v else False,
            "protection_basis": str(v.protection_basis.name) if v else "UNKNOWN",
            "theorem_reference": v.theorem_reference if v else "",
        },
        "momentum": {
            "total_diagrams": mc.total if mc else 0,
            "zero_transfer": len(mc.zero_transfer) if mc else 0,
            "nonzero_transfer": len(mc.nonzero_transfer) if mc else 0,
            "suppression_ratio": mc.suppression_ratio if mc else 0.0,
        },
        "topology": {
            "single_bubble": len(tc.single_bubble) if tc else 0,
            "ladder": len(tc.ladder) if tc else 0,
        },
        "errors": report.errors,
    }
    return result


def format_output(result: dict) -> str:
    """Format classification result for terminal display."""
    c = result["classification"]
    m = result["momentum"]
    t = result["topology"]

    protected = "YES [PASS]" if c["is_protected"] else "NO  [FAIL]"
    nonzero_val = "YES (injection term survives)" if c["matrix_element_nonzero"] else "NO (injection term vanishes)"

    lines = [
        "=" * 62,
        "  CGC -- Conservation-Guided Classification",
        "=" * 62,
        "",
        f"  Operator:      {result['operator_name']}",
        f"  Pipeline:      v{result['pipeline_version']}",
        f"  Timestamp:     {result['timestamp']}",
        "",
        "  ── Classification ──",
        f"  Protected:     {protected}",
        f"                   basis: {c['protection_basis']}",
        f"                   ref:   {c['theorem_reference']}",
        f"  Matrix element: {nonzero_val}",
        "",
        "  ── Diagram Analysis ──",
        f"  Total diagrams:     {m['total_diagrams']}",
        f"  Zero q-transfer:    {m['zero_transfer']}",
        f"  Nonzero q-transfer: {m['nonzero_transfer']}",
        f"  Suppression ratio:  {m['suppression_ratio']:.2f}",
        "",
        "  ── Topology ──",
        f"  Single-bubble:      {t['single_bubble']}",
        f"  Ladder (resummed):  {t['ladder']}",
        "",
    ]

    if result["errors"]:
        lines.append("  ── Errors ──")
        for err in result["errors"]:
            lines.append(f"  [!] {err}")
        lines.append("")

    # Emergence interpretation
    lines.append("  ── Emergence Interpretation ──")
    if c["is_protected"] and c["matrix_element_nonzero"]:
        lines.append("  --> Conserved operator: protected by symmetry theorem.")
        lines.append("  --> FRG trace density beta sign is determined by closed sector.")
        lines.append("  --> CGC classifies this as a DRIVING contribution to the RG flow.")
    else:
        lines.append("  --> Unprotected operator: injection term vanishes.")
        lines.append("  --> Pi0 determines whether this sector is IR-active or decoupled.")
        lines.append("  --> CGC classifies this as SPECTATOR (no direct conservation protection).")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CGC — Conservation-Guided Classification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  cgc-run --operator tmunu\n  cgc-run --operator f2 --output f2_result.json",
    )
    parser.add_argument(
        "--operator", "--op", "-o",
        dest="operator",
        choices=list(OPERATORS.keys()),
        help="Operator to classify",
    )
    parser.add_argument(
        "--list-operators", "-l",
        action="store_true",
        help="List available operators and exit",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Export full report to JSON file",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"cgc-run v{__import__('cgc').__version__}",
    )

    args = parser.parse_args()

    if args.list_operators:
        print("Available operators for cgc-run:")
        print()
        for key, entry in OPERATORS.items():
            conserved = "conserved [PASS]" if entry["is_conserved"] else "unprotected [FAIL]"
            print(f"  {key:12s}  {entry['name']}")
            print(f"              {conserved} — {entry['description']}")
            print()
        return

    if not args.operator:
        parser.error("--operator is required (use --list-operators to see options)")

    print(f"Running CGC classification for: {args.operator}")
    print()

    try:
        result = run_classification(args.operator)
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    print(format_output(result))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"Full report exported to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
