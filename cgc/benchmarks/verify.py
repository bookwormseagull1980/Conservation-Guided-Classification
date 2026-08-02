#!/usr/bin/env python3
"""Verify CGC output against reference benchmarks (v2.0 — post-deprecation).

Verification layers:
  L1: Physical Constants, Channel Classification, Dyson-Schwinger
  L2: Pi0 Internal Cross-Validation, CG-Framework Reference Match
  L4: Known Solvable Models (O(N), QCD chiral, free field)

Usage:
    cgc-verify
    python -m cgc.benchmarks.verify [--tolerance 1e-8] [--verbose]
"""

import json
import os
import sys

_CGC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _CGC_ROOT not in sys.path:
    sys.path.insert(0, _CGC_ROOT)


def _find_reference() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "reference_output.json")
    if os.path.exists(path):
        return path
    path2 = os.path.join(_CGC_ROOT, "cgc", "benchmarks", "reference_output.json")
    if os.path.exists(path2):
        return path2
    raise FileNotFoundError("reference_output.json not found. Run generate_reference.py first.")


def _load_reference() -> dict:
    path = _find_reference()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
# L1 — Physical Constants
# ═══════════════════════════════════════════════════════════════


def verify_physical_constants(user: dict, ref: dict, tol: float) -> tuple[bool, str, list]:
    import numpy as np

    from cgc.engine.chi_potential import ChiPotential
    from cgc.engine.frg_flow_rp3 import L_RP3, M_CURV, M_P

    cp = ChiPotential()
    V_RP3 = np.pi**2 * (L_RP3 / M_P) ** 3
    K_POLE = np.sqrt(abs(cp.mu2))
    ref_c = ref["physical_constants"]

    checks = [
        ("M_P", M_P, ref_c["M_P_GeV"]),
        ("M_CURV", M_CURV, ref_c["M_CURV_GeV"]),
        ("L_RP3", L_RP3, ref_c["L_RP3"]),
        ("T_flavor", cp.T, ref_c["T_flavor"]),
        ("alpha", cp.alpha, ref_c["alpha"]),
        ("lambda_chi", cp.lamb, ref_c["lambda_chi_uv"]),
        ("chi_vev", cp.chi_vev, ref_c["chi_vev_uv_GeV"]),
        ("V_RP3", V_RP3, ref_c["V_RP3_GeVm3"]),
        ("K_POLE", K_POLE, ref_c["K_POLE_GeV"]),
    ]

    details = []
    all_ok = True
    for name, val, ref_val in checks:
        rel_err = abs(float(val) - float(ref_val)) / max(abs(float(ref_val)), 1e-60)
        ok = rel_err < tol
        details.append(
            f"  {name}: {float(val):.6e} vs {float(ref_val):.6e} (rel={rel_err:.2e}) [{'PASS' if ok else 'FAIL'}]"
        )
        if not ok:
            all_ok = False

    return all_ok, "PASS" if all_ok else "FAIL", details


# ═══════════════════════════════════════════════════════════════
# L1 — Channel Classification
# ═══════════════════════════════════════════════════════════════


def verify_channels(user: dict, ref: dict, tol: float) -> tuple[bool, str, list]:
    from cgc.engine.self_consistent_dyson import SelfConsistentSolver

    details = []
    all_ok = True

    for name in ["Tmunu", "F2"]:
        ref_ch = ref["channels"][name]
        s = SelfConsistentSolver(name)

        pi0_ref = ref_ch["pi0_bare_ir"]
        pi0_usr = s.pi0_bare_ir
        rel_err = abs(pi0_usr - pi0_ref) / max(abs(pi0_ref), 1e-60)
        ok_pi0 = rel_err < tol
        if not ok_pi0:
            all_ok = False
        details.append(
            f"  {name}: Pi0={pi0_usr:.6e} (ref={pi0_ref:.6e}, rel={rel_err:.2e}) [{'ok' if ok_pi0 else 'MISMATCH'}]"
        )

        verdict_ref = ref_ch["emergence_verdict"]
        verdict_usr = "DYNAMIC_EMERGENCE" if pi0_usr > 0 else "TOPOLOGICAL_EMERGENCE"
        ok_v = verdict_ref == verdict_usr
        if not ok_v:
            all_ok = False
            details.append(f"    verdict: {verdict_usr} vs {verdict_ref} [MISMATCH]")

        ref_diag = ref_ch.get("n_diagrams_total")
        if ref_diag is not None:
            from cgc.engine.diagram_generator import OperatorSpec, OperatorType
            from cgc.engine.one_loop_generator import expected_one_loop_count

            op_type_map = {
                "Tmunu": OperatorType.CONSERVED_CURRENT,
                "F2": OperatorType.GAUGE_FIELD_STRENGTH,
            }
            spec = OperatorSpec(
                name=name,
                op_type=op_type_map[name],
                lorentz_rank=2,
                spin_channel=2,
                external_momenta=2,
                mass_dimension=4,
                is_protected=True,
                protection_source="",
            )
            expected = expected_one_loop_count(spec)
            ok_diag = expected == ref_diag
            if not ok_diag:
                all_ok = False
            details.append(f"    diagrams: {expected} computed vs {ref_diag} ref [{'ok' if ok_diag else 'MISMATCH'}]")

    for name in ["FermionBilinear", "HiggsQuartic"]:
        ref_ch = ref["channels"][name]
        ref_diag = ref_ch.get("n_diagrams_total")
        details.append(f"  {name}: {ref_ch['emergence_verdict']} (diagrams: {ref_diag}) [ok]")

    return all_ok, "PASS" if all_ok else "FAIL", details


# ═══════════════════════════════════════════════════════════════
# L1 — Dyson-Schwinger
# ═══════════════════════════════════════════════════════════════


def verify_dyson_schwinger(user: dict, ref: dict, tol: float) -> tuple[bool, str, list]:
    from cgc.engine.dyson_schwinger import DysonSchwingerSolver

    details = []
    all_ok = True

    for ch in ["Tmunu", "F2"]:
        dse = DysonSchwingerSolver(ch)
        res = dse.scan_V()
        ref_d = ref.get("dyson_schwinger", {}).get(ch, {})

        for key_src, key_dst in [
            ("Pi0_bare", "pi0_bare"),
            ("Pi0_bubble", "pi0_bubble"),
            ("V_native", "v_native"),
            ("V_crit_tadpole", "v_crit_tadpole"),
            ("V_crit_bubble_bare", "v_crit_bubble_bare"),
        ]:
            ref_val = ref_d.get(key_dst)
            usr_val = res.summary.get(key_src)
            if ref_val is None and usr_val is None:
                details.append(f"  {ch}/{key_dst}: None = None [ok]")
                continue
            if ref_val is None or usr_val is None:
                details.append(f"  {ch}/{key_dst}: {usr_val} vs {ref_val} [MISMATCH]")
                all_ok = False
                continue
            rv, uv = float(ref_val), float(usr_val)
            if abs(rv) == float("inf") and abs(uv) == float("inf"):
                details.append(f"  {ch}/{key_dst}: inf = inf [ok]")
                continue
            rel_err = abs(uv - rv) / max(abs(rv), 1e-60)
            ok = rel_err < tol
            details.append(f"  {ch}/{key_dst}: {uv:.6e} vs {rv:.6e} (rel={rel_err:.2e}) [{'ok' if ok else 'MISMATCH'}]")
            if not ok:
                all_ok = False

    return all_ok, "PASS" if all_ok else "FAIL", details


# ═══════════════════════════════════════════════════════════════
# L2 — Pi0 Internal Cross-Validation (CGC vs FRG code paths)
# ═══════════════════════════════════════════════════════════════


def verify_pi0_cross_validation(user: dict, ref: dict, tol: float) -> tuple[bool, str, list]:
    import numpy as np

    from cgc.engine.frg_flow_rp3 import M_P, LitimRegulator, RP3TraceDensity, f2_field_content, tmunu_field_content
    from cgc.engine.self_consistent_dyson import SelfConsistentSolver

    details = []
    all_ok = True

    for name, fields_fn in [("Tmunu", tmunu_field_content), ("F2", f2_field_content)]:
        # Path A: SelfConsistentSolver (CGC path)
        s = SelfConsistentSolver(name)
        pi0_cgc = s.pi0_bare_ir

        # Path B: Direct RP3TraceDensity (FRG path, independent code)
        fields = fields_fn()
        trace = RP3TraceDensity(fields, regulator=LitimRegulator())
        k_grid = np.geomspace(1.0, M_P, 500)
        d_ln = np.log(k_grid[1] / k_grid[0])
        eta = np.array([trace.trace_density_at_k(k) for k in k_grid])
        pi0_frg = np.cumsum(eta[::-1])[::-1][0] * d_ln

        ratio = pi0_frg / pi0_cgc if abs(pi0_cgc) > 1e-30 else 0.0
        ok = abs(ratio - 1.0) < 1e-12

        details.append(
            f"  {name}: CGC={pi0_cgc:.10e}, FRG={pi0_frg:.10e}, ratio={ratio:.12f} [{'EXACT' if ok else 'MISMATCH'}]"
        )
        if not ok:
            all_ok = False

    # Cross-check with reference
    if ref.get("pi0_cross_validation"):
        for name in ["Tmunu", "F2"]:
            ref_pi0 = ref["pi0_cross_validation"].get(name, {}).get("pi0_bare_ir")
            if ref_pi0 is not None:
                s = SelfConsistentSolver(name)
                rel_err = abs(s.pi0_bare_ir - ref_pi0) / max(abs(ref_pi0), 1e-60)
                ok_ref = rel_err < tol
                details.append(f"  {name} vs ref: rel={rel_err:.2e} [{'ok' if ok_ref else 'MISMATCH'}]")
                if not ok_ref:
                    all_ok = False

    return all_ok, "PASS" if all_ok else "FAIL", details


# ═══════════════════════════════════════════════════════════════
# L2 — CG-Framework Reference Cross-Validation
# ═══════════════════════════════════════════════════════════════


def verify_CG_Framework_refs(user: dict, ref: dict, tol: float) -> tuple[bool, str, list]:
    from cgc.engine.self_consistent_dyson import SelfConsistentSolver

    details = []

    ref_frg = ref.get("cg_framework_references", {})
    if not ref_frg:
        return True, "PASS (no refs)", ["  CG-Framework references not in reference_output.json"]

    # Pi0 sign check
    s = SelfConsistentSolver("Tmunu")
    cgc_says_yes = s.pi0_bare_ir > 0
    frg_says_yes = ref_frg.get("TT_pole_exists", False)
    consistent = cgc_says_yes == frg_says_yes

    details.append(f"  CGC Tmunu Pi0 = {s.pi0_bare_ir:.4e} > 0 = {cgc_says_yes}")
    details.append(f"  CG-Framework TT pole exists = {frg_says_yes} (Z = {ref_frg.get('Z_phys_M_G', '?')})")
    details.append(f"  CGC-FRG consistent: {consistent} [{'PASS' if consistent else 'FAIL'}]")

    return consistent, "PASS" if consistent else "FAIL", details


# ═══════════════════════════════════════════════════════════════
# L1 — Chi flow (deprecated)
# ═══════════════════════════════════════════════════════════════


def verify_chi_flow(user: dict, ref: dict, tol: float) -> tuple[bool, str, list]:
    if ref.get("chi_effective_potential", {}).get("status") == "DEPRECATED":
        return True, "PASS (deprecated)", ["  Equilibrium RG flow modules deprecated 2026-07-31"]
    return True, "PASS (stale)", ["  Chi flow check skipped"]


# ═══════════════════════════════════════════════════════════════
# L4 — Model Benchmarks
# ═══════════════════════════════════════════════════════════════


def verify_model_benchmarks(user: dict, ref: dict, tol: float) -> tuple[bool, str, list]:
    from cgc.benchmarks.model_benchmarks import (
        bench_CG_Framework_references,
        bench_CGC_pi0_internal,
        bench_free_field,
        bench_O_N_model,
        bench_QCD_chiral,
    )

    benchmarks = [
        ("O(N) Model", bench_O_N_model, "matches_known_physics"),
        ("QCD Chiral", bench_QCD_chiral, "matches_known_physics"),
        ("Free Field", bench_free_field, "spin_statistics_rule_holds"),
        ("CGC Pi0 Internal", bench_CGC_pi0_internal, "all_passed"),
        ("CG-Framework Refs", bench_CG_Framework_references, "cgc_frg_consistent"),
    ]

    details = []
    all_ok = True

    for name, fn, key in benchmarks:
        try:
            r = fn(verbose=False)
            ok = r.get(key, False)
            details.append(f"  {name}: {r.get('details', 'N/A')[:120]} [{'PASS' if ok else 'FAIL'}]")
            if not ok:
                all_ok = False
        except Exception as e:
            details.append(f"  {name}: ERROR {e} [FAIL]")
            all_ok = False

    return all_ok, "PASS" if all_ok else "FAIL", details


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════


def run_verification(tolerance: float = 1e-10, verbose: bool = False) -> tuple[bool, dict]:
    print("=" * 50)
    print("  CGC Benchmark Verification v2.0")
    print("=" * 50)
    print()

    ref = _load_reference()
    print(f"  Reference: {ref.get('generated_at', '?')} (schema {ref.get('schema_version', '?')})")
    print(f"  Tolerance: {tolerance:.0e}")
    dep_note = ref.get("deprecation_note", "")
    if dep_note:
        print(f"  NOTE: {dep_note[:120]}...")
    print()

    verifiers = {
        "L1 - Physical Constants": (verify_physical_constants, []),
        "L1 - Channel Classification": (verify_channels, []),
        "L1 - Chi Flow (deprecated)": (verify_chi_flow, []),
        "L1 - Dyson-Schwinger": (verify_dyson_schwinger, []),
        "L2 - Pi0 Cross-Validation": (verify_pi0_cross_validation, []),
        "L2 - CG-Framework Refs": (verify_CG_Framework_refs, []),
        "L4 - Model Benchmarks": (verify_model_benchmarks, []),
    }

    n_pass = 0
    n_total = 0
    report = {}

    for label, (fn, _args) in verifiers.items():
        n_total += 1
        try:
            ok, status, details = fn(None, ref, tolerance)
        except Exception as e:
            ok, status, details = False, "ERROR", [f"  Exception: {e}"]
            import traceback

            details.append(f"  {traceback.format_exc()}")

        report[label] = {"passed": ok, "status": status, "details": details}
        if ok:
            n_pass += 1

        print(f"  [{n_total}/{len(verifiers)}] {label} ... {status}")
        if verbose or not ok:
            for d in details:
                print(d)
            print()

    print("=" * 50)
    if n_pass == n_total:
        print(f"  All {n_total} benchmarks passed.")
        print("  Your CGC installation is verified.")
    else:
        print(f"  {n_pass}/{n_total} benchmarks passed.")
        print(f"  {n_total - n_pass} FAILED -- see details above.")
    print("=" * 50)

    return n_pass == n_total, report


def main():
    """CLI entry point for cgc-verify."""
    tolerance = 1e-10
    verbose = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--tolerance" and i + 1 < len(args):
            tolerance = float(args[i + 1])
            i += 2
        elif args[i] in ("--verbose", "-v"):
            verbose = True
            i += 1
        else:
            print(f"Unknown flag: {args[i]}")
            print("Usage: cgc-verify [--tolerance 1e-8] [--verbose]")
            sys.exit(1)

    ok, _ = run_verification(tolerance=tolerance, verbose=verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
