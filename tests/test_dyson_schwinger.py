"""Unit tests: Dyson-Schwinger and SelfConsistentSolver."""

from .conftest import RTOL_NUMERICAL, assert_close

from cgc.rp3_engine.dyson_schwinger import DysonSchwingerSolver
from cgc.rp3_engine.self_consistent_dyson import SelfConsistentSolver


def test_solver_Tmunu_pi0():
    """SelfConsistentSolver returns correct Tmunu Pi0."""
    s = SelfConsistentSolver("Tmunu")
    golden = 3.5999945350e-02
    assert_close(float(s.pi0_bare_ir), golden, rtol=RTOL_NUMERICAL, label="solver Tmunu pi0_bare_ir")


def test_solver_F2_pi0():
    """SelfConsistentSolver returns correct F2 Pi0."""
    s = SelfConsistentSolver("F2")
    golden = -1.5226901996e-01
    assert_close(float(s.pi0_bare_ir), golden, rtol=RTOL_NUMERICAL, label="solver F2 pi0_bare_ir")


def test_solver_Tmunu_v_crit():
    """Tmunu has a finite V_crit (positive Pi0 -> can bifurcate)."""
    s = SelfConsistentSolver("Tmunu")
    vc = s.find_v_crit()
    assert vc["found"], f"Tmunu should have V_crit, got {vc}"
    assert vc["v_crit"] > 0, f"Tmunu V_crit should be positive, got {vc['v_crit']}"
    assert vc["gap_decades"] > 0, "Tmunu gap_decades should be positive"
    assert_close(float(vc["v_crit"]), 4.1152325846, rtol=RTOL_NUMERICAL, label="Tmunu V_crit")


def test_solver_F2_no_v_crit():
    """F2 has NO V_crit (negative Pi0 -> never bifurcates)."""
    s = SelfConsistentSolver("F2")
    vc = s.find_v_crit()
    assert not vc["found"], f"F2 should NOT have V_crit (Pi0 < 0), got {vc}"
    assert vc["v_crit"] is None, "F2 V_crit should be None"


def test_solver_native_v():
    """Native V values are positive but much smaller than V_crit."""
    s_t = SelfConsistentSolver("Tmunu")
    s_f = SelfConsistentSolver("F2")
    assert s_t.native_v > 0
    assert s_f.native_v > 0
    assert s_t.native_v < s_t.find_v_crit()["v_crit"], (
        f"Tmunu V_native={s_t.native_v:.4e} >= V_crit 鈥?would be self-emerging"
    )


def test_dse_Tmunu():
    """Dyson-Schwinger solver produces valid scan results."""
    dse = DysonSchwingerSolver("Tmunu")
    res = dse.scan_V()
    assert res.summary["Pi0_bare"] > 0, "Pi0_bare should be positive"
    assert res.summary["V_native"] > 0


def test_dse_F2():
    """Dyson-Schwinger F2: negative Pi0 -> no tadpole crossing."""
    dse = DysonSchwingerSolver("F2")
    res = dse.scan_V()
    assert res.summary["Pi0_bare"] < 0, "F2 Pi0_bare should be negative"


def test_dse_results_consistent():
    """DSE results are internally consistent with solver."""
    for name in ["Tmunu", "F2"]:
        dse = DysonSchwingerSolver(name)
        res = dse.scan_V()
        s = SelfConsistentSolver(name)
        assert_close(
            float(res.summary["Pi0_bare"]), float(s.pi0_bare_ir), rtol=RTOL_NUMERICAL, label=f"DSEvsSolver {name} Pi0"
        )


def run_tests():
    import traceback

    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_") and callable(o)]
    n_pass = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}")
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL: {name} - {e}")
        except Exception as e:
            print(f"  ERROR: {name} - {e}")
            traceback.print_exc()
    print(f"\n  {n_pass}/{len(tests)} passed")
    return n_pass == len(tests)


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_tests() else 1)
