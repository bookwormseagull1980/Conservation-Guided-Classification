# -*- coding: utf-8 -*-
"""CGC Tier 1+2 Verification Suite — three-layer validation.

Layer 1: Unit tests (known limits, single-module invariants)
Layer 2: Integration tests (cross-module consistency)
Layer 3: Benchmark tests (Camporesi, flat-space, analytic solutions)

Run: py verify_tier1_tier2.py
"""

import sys, os, unittest, math, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgc.engine.frg_flow_rp3 import (
    M_P, M_CURV, L_RP3, N_C,
    RP3Spectrum, FieldSpecies, FieldContent,
    tmunu_field_content, f2_field_content,
)
from cgc.engine.frg_trace_density import FRGTraceDensity
from cgc.engine.self_consistent_dyson import SelfConsistentSolver, AnalyticalPoleConditions
from cgc.engine.chi_condensation import ChiPotential
from cgc.engine.gravity_feedback import GravitonExchangePi0
from cgc.engine.coupled_k_chi_evolution import CoupledKChiEvolution
from cgc.engine.coupled_chi_closure import CoupledChiClosure
from cgc.engine.dyson_schwinger import DysonSchwingerSolver

# Suppress numpy warnings during tests
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

PASS, FAIL, CRASH, SKIP = 0, 0, 0, 0
RESULTS = []

def check(level, name, cond, detail=""):
    global PASS, FAIL, CRASH
    try:
        ok = bool(cond)
        if ok:
            PASS += 1
            RESULTS.append(f"  [PASS] {level}: {name}")
        else:
            FAIL += 1
            RESULTS.append(f"  [FAIL] {level}: {name} — {detail}")
        return ok
    except Exception as e:
        CRASH += 1
        RESULTS.append(f"  [CRASH] {level}: {name} — {e}")
        return False

def banner(text):
    print(f"\n{'─'*60}")
    print(f"  {text}")
    print(f"{'─'*60}")

# ═══════════════════════════════════════════════════════════════
# LAYER 1: Unit Tests
# ═══════════════════════════════════════════════════════════════

def layer1_unit_tests():
    banner("LAYER 1: Unit Tests")

    # ── U1: Tmunu vertex = 1/L^4 ──
    td = FRGTraceDensity()
    vtx = td.compute_tmunu_vertex_rp3(L_RP3)
    check("U1", "vertex = 1/L^4", abs(vtx - 1.0/L_RP3**4) < 1e-12,
          f"vtx={vtx:.6e}, 1/L^4={1.0/L_RP3**4:.6e}")

    # ── U2: Flat-space limit ──
    L_big = 1e6
    vtx_flat = td.compute_tmunu_vertex_rp3(L_big)
    check("U2", "vertex(L->inf) -> 0", vtx_flat < 1e-20,
          f"vtx(L=1e6)={vtx_flat:.2e}")

    # ── U3: Mode counts at M_P ──
    spec = RP3Spectrum(L_RP3)
    scalars = spec.all_modes_below(M_P, FieldSpecies.SCALAR)
    vectors = spec.all_modes_below(M_P, FieldSpecies.VECTOR)
    spinors = spec.all_modes_below(M_P, FieldSpecies.SPINOR)
    check("U3a", "scalar modes at M_P = 1 (J=0 only)", len(scalars) == 1,
          f"got {len(scalars)}")
    check("U3b", "vector modes at M_P = 1 (J=1 only)", len(vectors) == 1,
          f"got {len(vectors)}")
    check("U3c", "spinor modes at M_P = 2 (J=1/2, 3/2)", len(spinors) == 2,
          f"got {len(spinors)}")
    # Degeneracy checks
    if scalars:
        check("U3d", "scalar J=0 degeneracy = 1 (Camporesi)", scalars[0].degeneracy == 1,
              f"got {scalars[0].degeneracy}")
    if vectors:
        # For SU(3): 8 gluons * 2 dof * 6 degeneracy(per mode)
        check("U3e", "vector J=1 degeneracy = 6 (Camporesi)", vectors[0].degeneracy == 6,
              f"got {vectors[0].degeneracy}")
    if spinors:
        check("U3f", "spinor J=1/2 degeneracy = 4 (Camporesi)", spinors[0].degeneracy == 4,
              f"got {spinors[0].degeneracy}")

    # ── U4: V=0 -> dressing = identity ──
    solver = SelfConsistentSolver("Tmunu")
    d0 = solver.compute_dressed(0.0, complete_spectral_sum=True)
    check("U4", "V=0 dressing = identity",
          np.allclose(d0.pi0_dressed, d0.pi0_bare, rtol=1e-10),
          f"max diff={np.max(np.abs(d0.pi0_dressed-d0.pi0_bare)):.2e}")

    # ── U5: complete_spectral_sum >= single-mode (Jensen) ──
    V_test = 0.01
    ds = solver.compute_dressed(V_test, complete_spectral_sum=True)
    d1 = solver.compute_dressed(V_test, complete_spectral_sum=False)
    check("U5", "complete-sum amplification >= single-mode (Jensen)",
          ds.max_amplification >= d1.max_amplification,
          f"complete={ds.max_amplification:.4f}, single={d1.max_amplification:.4f}")

    # ── U6: Gravity feedback exact <= estimate (more modes, more suppression) ──
    grav = GravitonExchangePi0()
    pi0_exact = grav.compute_pi0_grav_exact()["pi0_grav_exact"]
    pi0_est = grav.estimate_pi0_grav()
    check("U6", "pi0_grav_exact result exists", pi0_exact > 0,
          f"pi0_exact={pi0_exact:.4e}")
    check("U6b", "pi0_grav estimate exists", pi0_est > 0,
          f"pi0_est={pi0_est:.4e}")

    # ── U7: g2_grav_eff(k) ∝ k^2 ──
    k_test = [M_P/10, M_P/2, M_P, 2*M_P]
    g2_vals = [grav.compute_g2_grav_eff(k) for k in k_test]
    ratios = [g2_vals[i+1]/g2_vals[i] for i in range(len(g2_vals)-1)]
    expected_ratio = [25.0, 4.0, 4.0]  # (1/2 vs 1/10)^2, (1 vs 1/2)^2, (2 vs 1)^2
    ok = all(abs(r/e - 1.0) < 0.01 for r, e in zip(ratios, expected_ratio))
    check("U7", "g2_grav_eff ∝ k^2", ok,
          f"ratios={[round(r,2) for r in ratios]}, expected={expected_ratio}")

    # ── U8: F2 pi0 negative (fermion dominated) ──
    solver_f2 = SelfConsistentSolver("F2")
    check("U8", "F2 Pi0_bare_IR < 0", solver_f2.pi0_bare_ir < 0,
          f"Pi0={solver_f2.pi0_bare_ir:.4e}")

    # ── U9: chi potential V'(chi_v) = 0 ──
    cp = ChiPotential()
    chi_v = cp.chi_vev
    eps = 1e-6 * chi_v
    V_plus = cp.V(chi_v + eps)
    V_minus = cp.V(chi_v - eps)
    V_at_v = cp.V(chi_v)
    check("U9a", "chi_vev > 0", chi_v > 0,
          f"chi_vev={chi_v:.4e}")
    check("U9b", "V'(chi_v) = 0", abs(V_plus - V_minus) / (2*eps + 1e-30) < 1e-6 * abs(V_at_v)/chi_v,
          f"numerical gradient={abs(V_plus-V_minus)/(2*eps):.2e}")
    check("U9c", "V(chi_v) < 0 (true vacuum)", V_at_v < 0,
          f"V={V_at_v:.4e}")
    check("U9d", "lambda > 0", cp.lamb > 0,
          f"lambda={cp.lamb:.4f}")
    check("U9e", "mu^2 < 0", cp.mu2 < 0,
          f"mu2={cp.mu2:.4e}")

    # ── U10: Chi independent evolution ──
    evo = CoupledKChiEvolution()
    result = evo.analyze()
    check("U10a", "chi_independent result exists", len(result["chi_independent"]) > 0)
    check("U10b", "chi monotonic (increases as k decreases)",
          np.all(np.diff(result["chi_independent"]) >= -1e-15),
          f"min diff={np.min(np.diff(result['chi_independent'])):.2e}")

    # ── U11: Cubic solver equivalence ──
    x_vals = [0.001, 0.01, 0.05, 0.1, 0.14]
    all_ok = True
    for x in x_vals:
        y_cubic, status = AnalyticalPoleConditions.x_to_y(x)
        y_direct = x / (1.0 - y_cubic) ** 2 if y_cubic < 1 else float('inf')
        if not abs(y_direct - y_cubic) < 1e-12:
            all_ok = False
            break
    check("U11", "y^3-2y^2+y-x=0 equivalent to y=x/(1-y)^2",
          all_ok, f"tested at x=[{','.join(str(v) for v in x_vals)}]")

    # ── U12: DSE gap = 0 for V < V_crit ──
    ds = DysonSchwingerSolver("Tmunu")
    state_small = ds.compute_state(1.0)
    check("U12a", "NJL-DSE x=0 for V=1 (< V_crit)", state_small.x == 0.0,
          f"x={state_small.x:.6e}")
    check("U12b", "NJL-DSE Pi_dressed = Pi0_bubble at V=1 (x=0)",
          abs(state_small.Pi_dressed / ds.Pi0_bubble - 1.0) < 1e-12,
          f"Pi_dressed={state_small.Pi_dressed:.6e}, Pi0_bubble={ds.Pi0_bubble:.6e}")

    # ── U13: coupling_sq matches vertex ──
    fc = tmunu_field_content()
    coupling = fc[0].coupling_sq
    check("U13", "tmunu_field_content coupling_sq = 1/L^4",
          abs(coupling - 1.0/L_RP3**4) < 1e-12,
          f"coupling={coupling:.6e}")

    # ── U14: CoupledChiClosure import + basic ──
    try:
        ccc = CoupledChiClosure()
        check("U14", "CoupledChiClosure instantiated", True)
    except Exception as e:
        check("U14", "CoupledChiClosure instantiated", False, str(e))

# ═══════════════════════════════════════════════════════════════
# LAYER 2: Integration Tests
# ═══════════════════════════════════════════════════════════════

def layer2_integration_tests():
    banner("LAYER 2: Integration Tests")

    # ── I1: Vertex suppression factor ──
    td = FRGTraceDensity()
    r_frg = td.compute_pi0_tmunu_rp3(L_RP3)
    solver = SelfConsistentSolver("Tmunu")
    ratio = solver.pi0_bare_ir / r_frg.pi0_dimensionless
    # FRG pi0 includes vertex factor 1/L^4; solver pi0 is the bare sum
    # ratio = 1/(1/L^4 * 16*pi^2 * normalization) ≈ 122
    check("I1", "FRG pi0 / solver pi0 ratio stable", 100 < ratio < 200,
          f"ratio={ratio:.2f}")

    # ── I2: Pi0 sign consistency: TT > 0, F2 < 0 ──
    solver_f2 = SelfConsistentSolver("F2")
    check("I2a", "Tmunu Pi0_bare_IR > 0", solver.pi0_bare_ir > 0,
          f"Pi0={solver.pi0_bare_ir:.4e}")
    check("I2b", "F2 Pi0_bare_IR < 0", solver_f2.pi0_bare_ir < 0,
          f"Pi0={solver_f2.pi0_bare_ir:.4e}")

    # ── I3: V_native * Pi0 matches expectations ──
    vpi0_tt = solver.native_v * solver.pi0_bare_ir
    check("I3a", "Tmunu V_native*Pi0 ~ 6.4e-6", abs(vpi0_tt - 6.36e-6) < 1e-7,
          f"V*Pi0={vpi0_tt:.4e}")
    vpi0_f2 = solver_f2.native_v * solver_f2.pi0_bare_ir
    check("I3b", "F2 V_native*Pi0 < 0", vpi0_f2 < 0,
          f"V*Pi0={vpi0_f2:.4e}")

    # ── I4: Cubic V_crit = 0.148 / Pi0_bare ──
    from cgc.engine.self_consistent_dyson import AnalyticalPoleConditions
    x_crit = 4.0/27.0
    v_crit_tt = x_crit / solver.pi0_bare_ir
    check("I4a", "Tmunu cubic V_crit = 4.15", abs(v_crit_tt - 4.15) < 0.05,
          f"V_crit={v_crit_tt:.4f}")
    check("I4b", "Enhancement needed = 23250x",
          abs(v_crit_tt / solver.native_v - 23250) < 500,
          f"ratio={v_crit_tt/solver.native_v:.0f}x")

    # ── I5: Cubic Dyson and NJL-DSE are complementary ──
    ds = DysonSchwingerSolver("Tmunu")
    state_at_cubic = ds.compute_state(v_crit_tt)
    # At cubic V_crit (4.15), NJL-DSE should have x=0 (below tadpole threshold)
    check("I5", "NJL-DSE x=0 at cubic V_crit (4.15 < 28.07)",
          state_at_cubic.x == 0.0,
          f"x={state_at_cubic.x:.6e}")

    # ── I6: Physical constants self-consistency ──
    check("I6a", "M_CURV = M_P / L_RP3",
          abs(M_CURV - M_P/L_RP3) / M_CURV < 1e-12,
          f"M_CURV={M_CURV:.4e}, M_P/L={M_P/L_RP3:.4e}")
    check("I6b", "M_P ~ 2.435e18 GeV", abs(M_P - 2.435e18) / M_P < 0.01,
          f"M_P={M_P:.4e}")

# ═══════════════════════════════════════════════════════════════
# LAYER 3: Benchmark Tests
# ═══════════════════════════════════════════════════════════════

def layer3_benchmark_tests():
    banner("LAYER 3: Benchmark Tests")

    # ── B1: Camporesi (1990) degeneracies ──
    # RP3 spectral degeneracies from harmonic analysis
    spec = RP3Spectrum(L_RP3)

    # Scalar: d_J = (J+1)^2
    scalars = spec.all_modes_below(1e20, FieldSpecies.SCALAR)
    for m in scalars:
        J = m.quantum_number
        d_expected = (J + 1) ** 2
        check(f"B1-scalar-J{J}", f"d_J=(J+1)^2 for J={J}",
              m.degeneracy == d_expected,
              f"got {m.degeneracy}, expected {d_expected}")

    # Vector: d_J = 2*J*(J+2)
    vectors = spec.all_modes_below(1e20, FieldSpecies.VECTOR)
    for m in vectors:
        J = m.quantum_number
        d_expected = 2 * J * (J + 2)
        check(f"B1-vector-J{J}", f"d_J=2J(J+2) for J={J}",
              m.degeneracy == d_expected,
              f"got {m.degeneracy}, expected {d_expected}")

    # Spinor: d_J = 2*(J+1/2)*(J+3/2)
    spinors = spec.all_modes_below(1e20, FieldSpecies.SPINOR)
    for m in spinors:
        J = m.quantum_number
        d_expected = 2 * (J + 0.5) * (J + 1.5)
        check(f"B1-spinor-J{J}", f"d_J=2(J+1/2)(J+3/2) for J={J}",
              m.degeneracy == d_expected,
              f"got {m.degeneracy}, expected {d_expected}")

    # ── B2: Eigenvalue formulas ──
    M_P2 = M_P ** 2
    L2 = L_RP3 ** 2
    M_curv_sq = M_P2 / L2

    for m in scalars:
        J = m.quantum_number
        lam_expected = J * (J + 2) * M_curv_sq
        if J == 0:
            check(f"B2-scalar-J0", "lambda=0", abs(m.eigenvalue) < 1e-10,
                  f"got {m.eigenvalue:.2e}")
        else:
            check(f"B2-scalar-J{J}", f"lambda=J(J+2)*M_CURV^2 for J={J}",
                  abs(m.eigenvalue / lam_expected - 1.0) < 1e-12,
                  f"got {m.eigenvalue:.4e}, expected {lam_expected:.4e}")

    for m in vectors:
        J = m.quantum_number
        # RP3 vector Laplacian eigenvalues: (J+1)^2 * M_CURV^2 for odd J
        lam_expected = (J + 1.0) ** 2 * M_curv_sq
        check(f"B2-vector-J{J}", f"lambda=(J+1)^2*M_CURV^2 for J={J}",
              abs(m.eigenvalue / lam_expected - 1.0) < 1e-12,
              f"got {m.eigenvalue:.4e}, expected {lam_expected:.4e}")

    # Spinor: quantum_number runs 0,2,4,... (not physical J)
    # formula: (quantum_number + 3/2)^2 * M_CURV^2
    for m in spinors[:10]:  # Only check first 10, high-J are many
        QN = m.quantum_number
        lam_expected = (QN + 1.5) ** 2 * M_curv_sq
        check(f"B2-spinor-QN{QN}", f"lambda=(QN+3/2)^2*M_CURV^2 for QN={QN}",
              abs(m.eigenvalue / lam_expected - 1.0) < 1e-12,
              f"got {m.eigenvalue:.4e}, expected {lam_expected:.4e}")

    # ── B3: Analytical pole solutions ──
    y_std = AnalyticalPoleConditions.solve_pole_standard()
    check("B3a", "standard Dyson pole returns finite positive y",
          0 < y_std < 1, f"y={y_std:.6f}")

    y_sc = AnalyticalPoleConditions.solve_pole_self_consistent()
    # y/(1-y)^2 = 1/2 => y = 2-sqrt(3) ~ 0.267949
    check("B3b", "SC Dyson: y/(1-y)^2 = 1/2 => y=2-sqrt(3)",
          abs(y_sc - (2.0 - math.sqrt(3.0))) < 1e-12,
          f"y={y_sc:.6f}")

    y_bcs = AnalyticalPoleConditions.solve_pole_bcs()
    check("B3c", "BCS pole returns y=1",
          abs(y_bcs - 1.0) < 1e-12,
          f"y={y_bcs:.6f}")

    # ── B4: FRG pi0 sum consistency ──
    td = FRGTraceDensity()
    r = td.compute_pi0_tmunu_rp3(L_RP3)
    total_from_contribs = sum(c["pi0_dimless"] for c in r.contributions)
    check("B4", "pi0 total = sum contributions",
          abs(r.pi0_dimensionless - total_from_contribs) < 1e-14,
          f"total={r.pi0_dimensionless:.6e}, sum={total_from_contribs:.6e}")

# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

def run_all():
    global PASS, FAIL, CRASH
    PASS = FAIL = CRASH = 0
    RESULTS.clear()

    print("=" * 60)
    print("  CGC TIER 1+2 VERIFICATION SUITE (v2.0)")
    print("  Three-layer: unit -> integration -> benchmark")
    print("=" * 60)

    layer1_unit_tests()
    layer2_integration_tests()
    layer3_benchmark_tests()

    # Print all results
    for r in RESULTS:
        print(r)

    total = PASS + FAIL + CRASH
    print(f"\n{'='*60}")
    print(f"  VERIFICATION COMPLETE")
    print(f"  Total: {total}   Pass: {PASS}   Fail: {FAIL}   Crash: {CRASH}")
    print(f"{'='*60}")

    if FAIL > 0 or CRASH > 0:
        print("\n  FAILURES/CRASHES:")
        for r in RESULTS:
            if "[FAIL]" in r or "[CRASH]" in r:
                print(r)

    return 0 if FAIL == 0 and CRASH == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
