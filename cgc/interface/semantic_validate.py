r"""CGC↔FRG Semantic Validation — Real-data cross-validation.

Connects the actual FRG flow solver output to the CGC↔FRG interface,
replacing simulated data with real numerical results.

Validates:
  1. Pi0 agreement: CGC SelfConsistentSolver vs FRG RP3TraceDensity
     (should agree exactly — same trace density, same spectrum)
  2. V flow: FRG beta function integration → V_IR, log enhancement
  3. Cross-check: does FRG flow direction match CGC emergence verdict?
  4. Exports real FRG→CGC payload with actual flow data

Run:
  python -m cgc.interface.semantic_validate
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from cgc.rp3_engine.frg_flow_rp3 import (
    M_CURV,
    M_P,
    FlowConfig,
    RP3FRGFlowSolver,
    f2_field_content,
    tmunu_field_content,
)
from cgc.rp3_engine.self_consistent_dyson import SelfConsistentSolver
from cgc.interface.bridge import (
    CGC_FRG_Validator,
    CGCToFRGBridge,
    FRGToCGCBridge,
)
from cgc.interface.schema import (
    SCHEMA_VERSION,
)

# ═══════════════════════════════════════════════════════════════
# Core: Pi0 Validation
# ═══════════════════════════════════════════════════════════════


def validate_pi0_agreement(
    op_name: str,
    fields,
    native_v: float,
    n_k: int = 200,
) -> dict:
    """Verify that Pi0 computed by CGC and FRG paths agrees.

    CGC path:  SelfConsistentSolver._compute_pi0_grid()
    FRG path:  RP3FRGFlowSolver.compute_I(k) [which uses RP3TraceDensity]

    These should agree exactly since both use the same:
      - RP3TraceDensity(fields)
      - RP3Spectrum
      - Camporesi eigenvalue formulas
      - Litim threshold 2k^2/(k^2+m^2)

    Returns dict with comparison metrics.
    """
    # CGC path
    solver_cgc = SelfConsistentSolver(op_name)
    # Recompute at each scale
    k_grid_cgc = solver_cgc._k_grid
    pi0_grid_cgc = solver_cgc._pi0_grid

    # FRG path: use the trace density directly
    cfg_frg = FlowConfig(
        operator_name=op_name,
        v_uv=native_v,
        lambda_crit=1.0,  # dummy, not used here
        n_grid=n_k,
        include_rp3=True,
    )
    solver_frg = RP3FRGFlowSolver(cfg_frg)

    # Compute Pi0 from FRG trace density the same way CGC does:
    # Pi0(k) = ∫_{ln k}^{ln k_UV} d(ln p) eta(p)
    k_grid_frg = solver_frg._k_grid  # IR → UV
    eta_frg = np.array([solver_frg._trace.trace_density_at_k(k) for k in k_grid_frg])
    d_ln_frg = np.log(k_grid_frg[1] / k_grid_frg[0])
    partial_frg = np.cumsum(eta_frg) * d_ln_frg
    total_frg = partial_frg[-1]
    pi0_grid_frg = total_frg - partial_frg  # pi0[0]=total(at IR), pi0[-1]=0(at UV)

    # Cross-check at key scales
    key_scales = {
        "M_CURV": M_CURV,
        "M_G": solver_cgc.M_G if hasattr(solver_cgc, "M_G") else M_P / 1.438,
        "1 TeV": 1e3,
        "1 GeV": 1.0,
    }

    comparisons = {}
    for label, k_val in key_scales.items():
        idx_cgc = np.searchsorted(k_grid_cgc, k_val)
        idx_cgc = min(idx_cgc, len(k_grid_cgc) - 1)
        idx_frg = np.searchsorted(k_grid_frg, k_val)
        idx_frg = min(idx_frg, len(k_grid_frg) - 1)

        pc = float(pi0_grid_cgc[idx_cgc])
        pf = float(pi0_grid_frg[idx_frg])
        ratio = abs(pf / pc) if abs(pc) > 1e-30 else 0.0
        comparisons[label] = {
            "k_GeV": float(k_val),
            "pi0_cgc": pc,
            "pi0_frg": pf,
            "ratio": ratio,
            "match": bool(abs(ratio - 1.0) < 1e-6),  # should be exact
        }

    # IR Pi0 (the key number for emergence)
    pi0_ir_cgc = float(pi0_grid_cgc[0])
    pi0_ir_frg = float(pi0_grid_frg[0])

    return {
        "pi0_bare_ir_cgc": pi0_ir_cgc,
        "pi0_bare_ir_frg": pi0_ir_frg,
        "pi0_bare_ir_ratio": abs(pi0_ir_frg / pi0_ir_cgc) if abs(pi0_ir_cgc) > 1e-30 else 0.0,
        "pi0_match_exact": bool(abs(pi0_ir_frg - pi0_ir_cgc) < 1e-12),
        "comparisons_at_key_scales": comparisons,
        "notes": [
            f"Pi0_CGC(IR) = {pi0_ir_cgc:.6e}",
            f"Pi0_FRG(IR) = {pi0_ir_frg:.6e}",
            f"Ratio = {abs(pi0_ir_frg / pi0_ir_cgc) if abs(pi0_ir_cgc) > 1e-30 else 0:.10f}",
            "Expected: exact agreement (same RP3TraceDensity, same spectrum)",
        ],
    }


# ═══════════════════════════════════════════════════════════════
# Core: V Flow Validation
# ═══════════════════════════════════════════════════════════════


def validate_v_flow(
    op_name: str,
    fields,
    native_v: float,
    v_crit: float,
    n_grid: int = 500,
) -> dict:
    """Run FRG flow with real CGC parameters and report results.

    The FRG beta function flow integrates V(k) from UV to IR.
    This reports:
      - V_IR / V_UV enhancement
      - Whether V crosses V_crit during the flow
      - Beta function sign profile
    """
    cfg = FlowConfig(
        operator_name=op_name,
        v_uv=native_v,
        lambda_crit=v_crit,  # CGC-determined critical V
        n_grid=n_grid,
        include_rp3=True,
        include_anomalous_dim=True,
        include_v3=True,
    )
    solver = RP3FRGFlowSolver(cfg)
    flow = solver.solve()

    # Beta sign analysis
    beta_signs = np.sign(flow.beta_grid)
    n_sign_flips = int(sum(1 for i in range(1, len(beta_signs)) if beta_signs[i] != beta_signs[i - 1]))

    # I(k) profile at key scales
    key_I = {}
    for k_label, k_val in [
        ("M_P", M_P),
        ("M_CURV", M_CURV),
        ("1 TeV", 1e3),
        ("1 GeV", 1.0),
    ]:
        key_I[k_label] = float(solver.compute_I(k_val))

    return {
        "v_uv": float(flow.v_uv),
        "v_ir": float(flow.v_ir),
        "log_enhancement": float(flow.log_enhancement),
        "linear_enhancement": float(flow.v_ir / flow.v_uv) if flow.v_uv > 0 else 0.0,
        "crosses_critical": bool(flow.crosses_critical),
        "k_cross_GeV": float(flow.k_cross) if flow.k_cross else None,
        "v_crit": v_crit,
        "gap_to_criticality": float(v_crit / flow.v_ir) if flow.v_ir > 0 else float("inf"),
        "beta_sign_flips": n_sign_flips,
        "beta_sign_at_ir": int(beta_signs[0]) if len(beta_signs) > 0 else 0,
        "beta_sign_at_uv": int(beta_signs[-1]) if len(beta_signs) > 0 else 0,
        "I_at_key_scales": key_I,
        "flow_result": flow,  # raw FlowResult for payload building
        "notes": flow.notes.copy(),
    }


# ═══════════════════════════════════════════════════════════════
# Full Semantic Validation Runner
# ═══════════════════════════════════════════════════════════════


def run_semantic_validation(
    output_dir: str = None,
) -> dict:
    """Complete semantic validation: CGC↔FRG round-trip with real data.

    Steps:
      1. Run CGC SelfConsistentSolver → operator spec, Pi0, V_native
      2. Run FRG flow solver → V(k) trajectory, Pi0 from trace density
      3. Cross-validate Pi0 (exact agreement expected)
      4. Cross-validate V flow vs CGC emergence prediction
      5. Export real CGC→FRG and FRG→CGC payloads
      6. Generate validation report
    """
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parents[2] / "output")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  CGC↔FRG SEMANTIC VALIDATION — REAL DATA")
    print("=" * 72)

    results = {}

    # ═══════ Tmunu ═══════
    print("\n" + "─" * 72)
    print("  TMUNU (TT PROJECTION, SPIN-2) — CONSERVED CURRENT")
    print("─" * 72)

    # Step 1: CGC
    solver_tmunu = SelfConsistentSolver("Tmunu")
    native_v_tmunu = solver_tmunu.native_v
    pi0_cgc_tmunu = solver_tmunu.pi0_bare_ir
    lambda_crit = 4.0 / 27.0  # cubic bifurcation
    v_crit_tmunu = lambda_crit / pi0_cgc_tmunu if pi0_cgc_tmunu > 0 else float("inf")

    print(f"\n  [CGC] native_v  = {native_v_tmunu:.6e}")
    print(f"  [CGC] Pi0(IR)   = {pi0_cgc_tmunu:.6e}")
    print(f"  [CGC] V_crit    = {v_crit_tmunu:.4f}  (lambda_crit = 4/27 = {lambda_crit:.6f})")
    print(f"  [CGC] Gap       = x{v_crit_tmunu / native_v_tmunu:.1f}")
    print("  [CGC] Verdict   = DYNAMIC_EMERGENCE (Pi0>0, protected)")

    # BCS analysis
    bcs = solver_tmunu.find_bcs_critical_temperature() if hasattr(solver_tmunu, "find_bcs_critical_temperature") else {}
    has_bcs = bcs.get("has_bcs_solution", False)
    bcs_integral = bcs.get("bcs_integral_at_native_V", None)

    # Step 2: Pi0 cross-validation
    print("\n  ── Pi0 Cross-Validation ──")
    pi0_val = validate_pi0_agreement("Tmunu", tmunu_field_content(), native_v_tmunu)

    print(f"  CGC Pi0(IR) = {pi0_val['pi0_bare_ir_cgc']:.8e}")
    print(f"  FRG Pi0(IR) = {pi0_val['pi0_bare_ir_frg']:.8e}")
    match_str = "✅ EXACT" if pi0_val["pi0_match_exact"] else "❌ MISMATCH"
    print(f"  Agreement:   {match_str}  (ratio={pi0_val['pi0_bare_ir_ratio']:.10f})")

    for label, comp in pi0_val["comparisons_at_key_scales"].items():
        print(
            f"    {label:>8s} @ {comp['k_GeV']:.2e} GeV: "
            f"CGC={comp['pi0_cgc']:+.6e}, FRG={comp['pi0_frg']:+.6e}, "
            f"ratio={comp['ratio']:.10f} "
            f"{'✅' if comp['match'] else '⚠️'}"
        )

    # Step 3: V flow
    print("\n  ── V Flow Integration ──")
    flow_val = validate_v_flow(
        "Tmunu",
        tmunu_field_content(),
        native_v_tmunu,
        v_crit_tmunu,
        n_grid=500,
    )

    print(f"  V_UV          = {flow_val['v_uv']:.6e}")
    print(f"  V_IR          = {flow_val['v_ir']:.6e}")
    print(f"  log enhance   = {flow_val['log_enhancement']:+.4f}")
    print(f"  linear enh    = x{flow_val['linear_enhancement']:.4f}")
    print(f"  V_crit        = {flow_val['v_crit']:.4f}")
    print(
        f"  Cross crit?   = {'✅ YES' if flow_val['crosses_critical'] else '❌ NO (expected: gap x23275, FRG alone insufficient)'}"
    )
    print(f"  Gap to crit   = x{flow_val['gap_to_criticality']:.1f}")
    print(f"  Beta signflips= {flow_val['beta_sign_flips']}")
    print(f"  Beta(IR)      = {'+' if flow_val['beta_sign_at_ir'] > 0 else '-'}")
    print(f"  Beta(UV)      = {'+' if flow_val['beta_sign_at_uv'] > 0 else '-'}")
    for k_label, I_val in flow_val["I_at_key_scales"].items():
        print(f"    I({k_label:>8s}) = {I_val:+.6e}")

    results["tmunu"] = {
        "pi0_validation": pi0_val,
        "flow_validation": flow_val,
    }

    # Step 4: Build CGC→FRG payload
    bridge = CGCToFRGBridge()
    cgc_payload = bridge.from_self_consistent_solver(
        operator_name="Tmunu (TT projection, spin-2)",
        field_content=tmunu_field_content(),
        pi0_bare_ir=pi0_cgc_tmunu,
        native_v=native_v_tmunu,
        is_protected=True,
        protection_basis="WARD_IDENTITY",
        op_type="CONSERVED_CURRENT",
        bcs_integral=bcs_integral,
        has_bcs=has_bcs,
    )
    cgc_payload.resummation.n_q0_diagrams = 3
    cgc_payload.resummation.n_ladder_diagrams = 0
    cgc_payload.resummation.n_single_bubble = 3
    cgc_payload.resummation.n_total_diagrams = 6

    cgc_path = f"{output_dir}/cgc_to_frg_tmunu_REAL.json"
    cgc_payload.export_json(cgc_path)
    print(f"\n  ✅ CGC→FRG exported: {cgc_path}")

    # Step 5: Build FRG→CGC payload (REAL data, not simulated)
    frg_bridge = FRGToCGCBridge()
    frg_payload = frg_bridge.from_rp3_flow_result(
        operator_name="Tmunu (TT projection, spin-2)",
        flow_result=flow_val["flow_result"],
        pi0_bare_ir=pi0_cgc_tmunu,
        pi0_frg_computed=pi0_val["pi0_bare_ir_frg"],
        cgc_payload_path=cgc_path,
    )

    frg_path = f"{output_dir}/frg_to_cgc_tmunu_REAL.json"
    frg_payload.export_json(frg_path)
    print(f"  ✅ FRG→CGC exported: {frg_path}")

    # Step 6: Cross-validate
    xval = CGC_FRG_Validator.validate(cgc_payload, frg_payload)
    val_path = f"{output_dir}/validation_tmunu_REAL.json"

    val_report = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": cgc_payload.timestamp,
        "operator": "Tmunu (TT projection, spin-2)",
        "cgc_payload": cgc_path,
        "frg_payload": frg_path,
        "pi0_validation": {
            "pi0_cgc": pi0_val["pi0_bare_ir_cgc"],
            "pi0_frg": pi0_val["pi0_bare_ir_frg"],
            "ratio": pi0_val["pi0_bare_ir_ratio"],
            "exact_match": pi0_val["pi0_match_exact"],
        },
        "flow_validation": {
            "v_uv": flow_val["v_uv"],
            "v_ir": flow_val["v_ir"],
            "log_enhancement": flow_val["log_enhancement"],
            "crosses_critical": flow_val["crosses_critical"],
            "v_crit": flow_val["v_crit"],
            "gap_to_criticality": flow_val["gap_to_criticality"],
            "beta_sign_flips": flow_val["beta_sign_flips"],
        },
        "cross_validation": xval.to_dict(),
        "verdict_match": xval.verdict_match,
        "agreement_level": xval.agreement_level,
        "pi0_match": xval.pi0_match,
        "notes": list(xval.notes),
    }
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Validation report: {val_path}")

    results["tmunu"]["cgc_payload"] = cgc_payload
    results["tmunu"]["frg_payload"] = frg_payload
    results["tmunu"]["cross_validation"] = xval

    # ═══════ F2 ═══════
    print("\n" + "─" * 72)
    print("  F² (GAUGE FIELD STRENGTH, SU(3)) — BRST-PROTECTED")
    print("─" * 72)

    solver_f2 = SelfConsistentSolver("F2")
    native_v_f2 = solver_f2.native_v
    pi0_cgc_f2 = solver_f2.pi0_bare_ir

    print(f"\n  [CGC] native_v  = {native_v_f2:.6e}")
    print(f"  [CGC] Pi0(IR)   = {pi0_cgc_f2:.6e}")
    print(f"  [CGC] Pi0 sign  = {'NEGATIVE → fermion-dominated → Dyson suppression' if pi0_cgc_f2 < 0 else 'POSITIVE'}")
    print("  [CGC] Verdict   = TOPOLOGICAL_EMERGENCE (protected but Pi0<0)")
    print("  [CGC] Mechanism = pole from equivariant index theorem, not ladder resummation")

    # Pi0 cross-validation
    print("\n  ── Pi0 Cross-Validation ──")
    pi0_val_f2 = validate_pi0_agreement("F2", f2_field_content(), native_v_f2)
    print(f"  CGC Pi0(IR) = {pi0_val_f2['pi0_bare_ir_cgc']:.8e}")
    print(f"  FRG Pi0(IR) = {pi0_val_f2['pi0_bare_ir_frg']:.8e}")
    match_str = "✅ EXACT" if pi0_val_f2["pi0_match_exact"] else "❌ MISMATCH"
    print(f"  Agreement:   {match_str}")

    # V flow (F2 has Pi0<0, so V_crit concept doesn't apply here;
    # lambda_crit used only for flow integration, set to large dummy)
    print("\n  ── V Flow Integration ──")
    flow_val_f2 = validate_v_flow(
        "F2",
        f2_field_content(),
        native_v_f2,
        28.0,  # original lambda_crit from frg_flow_rp3
        n_grid=500,
    )
    print(f"  V_UV          = {flow_val_f2['v_uv']:.6e}")
    print(f"  V_IR          = {flow_val_f2['v_ir']:.6e}")
    print(f"  log enhance   = {flow_val_f2['log_enhancement']:+.4f}")
    print(f"  Cross crit?   = {'✅ YES' if flow_val_f2['crosses_critical'] else '❌ NO'}")

    # CGC→FRG payload
    cgc_payload_f2 = bridge.from_self_consistent_solver(
        operator_name="F^2 (gauge field strength, SU(3))",
        field_content=f2_field_content(),
        pi0_bare_ir=pi0_cgc_f2,
        native_v=native_v_f2,
        is_protected=True,
        protection_basis="BRST_SYMMETRY",
        op_type="GAUGE_FIELD_STRENGTH",
    )
    cgc_path_f2 = f"{output_dir}/cgc_to_frg_f2_REAL.json"
    cgc_payload_f2.export_json(cgc_path_f2)

    # FRG→CGC payload
    frg_payload_f2 = frg_bridge.from_rp3_flow_result(
        operator_name="F^2 (gauge field strength, SU(3))",
        flow_result=flow_val_f2["flow_result"],
        pi0_bare_ir=pi0_cgc_f2,
        pi0_frg_computed=pi0_val_f2["pi0_bare_ir_frg"],
        cgc_payload_path=cgc_path_f2,
    )
    frg_path_f2 = f"{output_dir}/frg_to_cgc_f2_REAL.json"
    frg_payload_f2.export_json(frg_path_f2)

    # Cross-validate
    xval_f2 = CGC_FRG_Validator.validate(cgc_payload_f2, frg_payload_f2)
    val_path_f2 = f"{output_dir}/validation_f2_REAL.json"
    with open(val_path_f2, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "operator": "F^2",
                "cross_validation": xval_f2.to_dict(),
                "verdict_match": xval_f2.verdict_match,
                "agreement_level": xval_f2.agreement_level,
                "notes": list(xval_f2.notes),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n  ✅ Validation report: {val_path_f2}")

    results["f2"] = {
        "pi0_validation": pi0_val_f2,
        "flow_validation": flow_val_f2,
        "cgc_payload": cgc_payload_f2,
        "frg_payload": frg_payload_f2,
        "cross_validation": xval_f2,
    }

    # ═══════ Summary ═══════
    print(f"\n{'=' * 72}")
    print("  SEMANTIC VALIDATION SUMMARY")
    print(f"{'=' * 72}")

    for op in ["tmunu", "f2"]:
        r = results[op]
        pi0 = r["pi0_validation"]
        flow = r["flow_validation"]
        xval = r["cross_validation"]

        print(f"\n  [{op.upper()}]")
        print(
            f"    Pi0: CGC={pi0['pi0_bare_ir_cgc']:.4e}, "
            f"FRG={pi0['pi0_bare_ir_frg']:.4e}, "
            f"match={'✅' if pi0['pi0_match_exact'] else '❌'}"
        )
        print(f"    V(UV→IR): {flow['v_uv']:.4e}→{flow['v_ir']:.4e}, log_enh={flow['log_enhancement']:+.4f}")
        print(f"    Cross-validate: {xval.agreement_level}, verdict_match={'✅' if xval.verdict_match else '❌'}")

    return results


if __name__ == "__main__":
    run_semantic_validation()
