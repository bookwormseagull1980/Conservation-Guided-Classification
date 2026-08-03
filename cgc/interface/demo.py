"""CGC↔FRG Interface Demo — End-to-end round-trip with real data.

Demonstrates:
  1. Build CGC→FRG payload from SelfConsistentSolver results
  2. Export to JSON
  3. Import back and validate
  4. Simulate FRG→CGC response
  5. Cross-validate CGC prediction vs FRG result
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add engines to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cgc.interface.bridge import (
    CGCToFRGBridge,
    FRGToCGCBridge,
    import_frg_to_cgc,
)
from cgc.interface.schema import (
    SCHEMA_VERSION,
    CGCToFRGPayload,
    EmergenceVerdict,
    get_json_schema,
)


def build_tmunu_payload() -> CGCToFRGPayload:
    """Build CGC→FRG payload for Tmunu using real CG-Framework numbers."""
    from cgc.rp3_engine.frg_flow_rp3 import tmunu_field_content
    from cgc.rp3_engine.self_consistent_dyson import SelfConsistentSolver

    # Get results from self-consistent solver
    solver = SelfConsistentSolver("Tmunu")
    native_v = solver.native_v
    pi0_ir = solver.pi0_bare_ir

    print(f"Tmunu: native_v = {native_v:.4e}, pi0_bare_ir = {pi0_ir:.4e}")

    # BCS-type analysis
    bcs = solver.find_bcs_critical_temperature()
    bcs_integral = bcs.get("bcs_integral_at_native_V", None)
    has_bcs = bcs.get("has_bcs_solution", False)

    print(f"BCS integral at native V: {bcs_integral:.4e}" if bcs_integral else "BCS: not computed")
    print(f"Has BCS solution: {has_bcs}")

    # Build payload
    bridge = CGCToFRGBridge()
    payload = bridge.from_self_consistent_solver(
        operator_name="Tmunu (TT projection, spin-2)",
        field_content=tmunu_field_content(),
        pi0_bare_ir=pi0_ir,
        native_v=native_v,
        is_protected=True,
        protection_basis="WARD_IDENTITY",
        op_type="CONSERVED_CURRENT",
        bcs_integral=bcs_integral,
        has_bcs=has_bcs,
    )

    # Add diagram counts (from CGC pipeline benchmark_tmunu_spin2.json)
    payload.resummation.n_q0_diagrams = 3
    payload.resummation.n_ladder_diagrams = 0  # single-bubble at 1-loop
    payload.resummation.n_single_bubble = 3
    payload.resummation.n_total_diagrams = 6

    return payload


def build_f2_payload() -> CGCToFRGPayload:
    """Build CGC→FRG payload for F² using real CG-Framework numbers."""
    from cgc.rp3_engine.frg_flow_rp3 import f2_field_content
    from cgc.rp3_engine.self_consistent_dyson import SelfConsistentSolver

    solver = SelfConsistentSolver("F2")
    native_v = solver.native_v
    pi0_ir = solver.pi0_bare_ir

    print(f"F2:   native_v = {native_v:.4e}, pi0_bare_ir = {pi0_ir:.4e}")

    bridge = CGCToFRGBridge()
    return bridge.from_self_consistent_solver(
        operator_name="F^2 (gauge field strength, SU(3))",
        field_content=f2_field_content(),
        pi0_bare_ir=pi0_ir,
        native_v=native_v,
        is_protected=True,
        protection_basis="BRST_SYMMETRY",
        op_type="GAUGE_FIELD_STRENGTH",
    )



def validate_example(payload: CGCToFRGPayload, tag: str) -> None:
    """Basic structural validation of a payload."""
    print(f"\n── {tag} Validation ──")
    assert payload.schema_version == SCHEMA_VERSION, "Schema version mismatch"
    assert payload.operator.name, "Missing operator name"
    assert len(payload.fields) > 0, "No field content"
    assert payload.spectrum.L_rp3 > 0, "Invalid L_rp3"
    assert payload.spectrum.M_P > 0, "Invalid M_P"

    # Check operator spec
    op = payload.operator
    print(f"  Operator: {op.name}")
    print(f"  Type: {op.operator_type.value}")
    print(f"  Protected: {op.is_protected}")
    print(f"  Pi0 sign: {'+' if op.pi0_sign > 0 else '-' if op.pi0_sign < 0 else '0'}")
    print(f"  Emergence: {op.emergence_verdict.value}")
    if op.pi0_bare_ir is not None:
        print(f"  Pi0_bare(IR): {op.pi0_bare_ir:.4e}")
    if op.v_native is not None:
        print(f"  V_native: {op.v_native:.4e}")
    if op.gap_to_criticality is not None:
        print(f"  Gap to critical: x{op.gap_to_criticality:.1f}")

    # Field content
    print(f"  Fields: {len(payload.fields)} species")
    for f in payload.fields[:5]:
        print(
            f"    {f.name:20s} {f.field_type.value:8s} "
            f"n={f.n_species} dof={f.dof_per_species} "
            f"m={f.mass_gev:.1f} c²={f.coupling_sq:.4e}"
        )

    # Instructions
    print(f"  Instructions: {len(payload.compute_instructions)} items")
    for ci in payload.compute_instructions[:2]:
        print(f"    → {ci}")


def demo_roundtrip():
    """Full round-trip demo."""
    output_dir = Path(__file__).resolve().parents[2] / "output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 72)
    print("  CGC ↔ FRG JSON INTERFACE — ROUND-TRIP DEMO")
    print("=" * 72)

    # ── Step 1: Build CGC→FRG payloads ──
    print("\n── Step 1: Building CGC→FRG payloads ──")
    tmunu_payload = build_tmunu_payload()
    f2_payload = build_f2_payload()

    # Validate
    validate_example(tmunu_payload, "Tmunu")
    validate_example(f2_payload, "F2")

    # ── Step 2: Export to JSON ──
    print("\n── Step 2: Exporting to JSON ──")
    tmunu_path = str(output_dir / "cgc_to_frg_tmunu.json")
    f2_path = str(output_dir / "cgc_to_frg_f2.json")

    tmunu_payload.export_json(tmunu_path)
    f2_payload.export_json(f2_path)
    print(f"  ✅ {tmunu_path}")
    print(f"  ✅ {f2_path}")

    # ── Step 3: Import back from JSON ──
    print("\n── Step 3: Re-importing from JSON ──")
    tmunu_loaded = CGCToFRGPayload.import_json(tmunu_path)
    f2_loaded = CGCToFRGPayload.import_json(f2_path)

    assert tmunu_loaded.operator.name == tmunu_payload.operator.name
    assert f2_loaded.operator.emergence_verdict == EmergenceVerdict.TOPOLOGICAL_EMERGENCE
    print(f"  ✅ Tmunu round-trip: {tmunu_loaded.operator.name}")
    print(f"  ✅ F2 round-trip: {f2_loaded.operator.emergence_verdict.value}")

    # ── Step 4: Build simulated FRG→CGC payload ──
    print("\n── Step 4: Simulating FRG→CGC response ──")
    bridge = FRGToCGCBridge()

    # Simulate a FlowResult-like object
    class SimFlowResult:
        crosses_critical = True
        v_uv = tmunu_payload.operator.v_native or 1.79e-4
        v_ir = tmunu_payload.operator.v_crit_needed or 3.0e-2
        log_enhancement = 5.2
        k_cross = 5e17
        notes = ["Simulated FRG flow for Tmunu"]
        k_grid = []  # simplified
        v_grid = []
        eta_grid = []

    frg_payload = bridge.from_rp3_flow_result(
        operator_name="Tmunu (TT projection, spin-2)",
        flow_result=SimFlowResult(),
        pi0_bare_ir=tmunu_payload.operator.pi0_bare_ir,
        pi0_frg_computed=tmunu_payload.operator.pi0_bare_ir * 1.05,
        cgc_payload_path=tmunu_path,
    )

    frg_path = str(output_dir / "frg_to_cgc_tmunu.json")
    frg_payload.export_json(frg_path)
    print(f"  ✅ {frg_path}")
    print(f"  Pole status: {frg_payload.spectral_function.pole_status.value}")

    # ── Step 5: Cross-validate ──
    print("\n── Step 5: Cross-Validation ──")
    result = import_frg_to_cgc(
        frg_json_path=frg_path,
        cgc_json_path=tmunu_path,
        validate=True,
        output_validation_path=str(output_dir / "validation_tmunu.json"),
    )

    xval = result["validation"]
    print(f"  Pi0 match:     {'✅' if xval.pi0_match else '❌'}")
    print(f"  Verdict match: {'✅' if xval.verdict_match else '❌'}")
    print(f"  Agreement:     {xval.agreement_level}")
    print("  Notes:")
    for n in xval.notes:
        print(f"    → {n}")

    # ── Step 6: Show JSON Schema ──
    print("\n── Step 6: JSON Schema (CGC→FRG) ──")
    schema = get_json_schema("cgc_to_frg")
    schema_path = str(output_dir / "cgc_to_frg_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"  ✅ Schema exported to {schema_path}")

    # ── Summary ──
    print("\n" + "=" * 72)
    print("  ROUND-TRIP SUMMARY")
    print("=" * 72)
    print("  CGC→FRG payloads:       tmunu + f2")
    print("  FRG→CGC payload:        tmunu (simulated)")
    print(f"  Cross-validation:       {xval.agreement_level}")
    print("  JSON Schema:            CGC→FRG + FRG→CGC (Draft-07)")
    print(f"  Output directory:       {output_dir}")
    print(f"  Schema version:         {SCHEMA_VERSION}")

    return {
        "tmunu": tmunu_payload,
        "f2": f2_payload,
        "frg_response": frg_payload,
        "validation": xval,
    }


if __name__ == "__main__":
    demo_roundtrip()
