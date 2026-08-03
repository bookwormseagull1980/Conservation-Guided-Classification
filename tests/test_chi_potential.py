"""Unit tests: ChiPotential 鈥?EC geometric constants.

These are the MOST FUNDAMENTAL tests: if ChiPotential's parameters
are wrong, everything downstream is garbage.

Tests:
  - EC geometry parameters are deterministic and traceable
  - chi_vev = M_P / sqrt(T-2)  (RP3 first Cartan invariant)
  - mu2 = -alpha * M_P^2        (Holst-deformed mass)
  - lambda = 6*alpha / (chi_vev/M_P)^2  (quartic self-coupling)
  - Full closure: inputs -> outputs, zero free parameters
"""

from .conftest import (
    ALPHA,
    M_P,
    T_FLAVOR,
    assert_close,
)


def test_chi_vev_derivation():
    """chi_vev = M_P / sqrt(T-2) from RP3 Cartan geometry."""
    import numpy as np

    chi_vev_expected = M_P / np.sqrt(T_FLAVOR - 2)  # T=5 -> 1/sqrt(3)
    from cgc.rp3_engine.chi_potential import ChiPotential

    cp = ChiPotential()
    assert_close(cp.chi_vev, chi_vev_expected, label="chi_vev")

    # Exact ratio match (no rounding)
    ratio = cp.chi_vev / M_P
    expected_ratio = 1.0 / np.sqrt(T_FLAVOR - 2)
    assert_close(ratio, expected_ratio, label="chi_vev/M_P")


def test_mu2_derivation():
    """mu2 = -alpha * M_P^2 from Holst-deformed mass."""
    mu2_expected = -ALPHA * M_P**2
    from cgc.rp3_engine.chi_potential import ChiPotential

    cp = ChiPotential()
    assert_close(cp.mu2, mu2_expected, label="mu2")


def test_lambda_derivation():
    """lambda = 6*alpha / (chi_vev/M_P)^2 from quartic self-coupling."""
    from cgc.rp3_engine.chi_potential import ChiPotential

    cp = ChiPotential()
    chi_ratio = cp.chi_vev / M_P
    lamb_expected = 6.0 * ALPHA / chi_ratio**2
    assert_close(cp.lamb, lamb_expected, label="lambda")


def test_T_preserved():
    """T flavor index is preserved exactly (integer input)."""
    from cgc.rp3_engine.chi_potential import ChiPotential

    cp = ChiPotential()
    assert isinstance(cp.T, int) or (cp.T % 1 == 0), f"T={cp.T} is not integer"
    assert cp.T == T_FLAVOR, f"T={cp.T} != {T_FLAVOR}"


def test_alpha_preserved():
    """alpha is preserved exactly (float input)."""
    from cgc.rp3_engine.chi_potential import ChiPotential

    cp = ChiPotential()
    assert cp.alpha == ALPHA, f"alpha={cp.alpha} != {ALPHA}"


def test_full_closure():
    """All 3 derived quantities are consistent with each other."""
    from cgc.rp3_engine.chi_potential import ChiPotential

    cp = ChiPotential()
    chi_ratio = cp.chi_vev / M_P

    # lambda = -6 * mu2 / (M_P^2 * chi_ratio^2)
    # Check: -6 * mu2 / (M_P^2 * chi_ratio^2) == lambda
    lhs = -6.0 * cp.mu2 / (M_P**2 * chi_ratio**2)
    assert_close(lhs, cp.lamb, label="lambda from mu2 closure")


def test_no_free_parameters():
    """ChiPotential takes NO arguments 鈥?zero free parameters."""
    from cgc.rp3_engine.chi_potential import ChiPotential

    cp1 = ChiPotential()
    cp2 = ChiPotential()
    # Two instances must give identical results
    for attr in ["T", "alpha", "chi_vev", "mu2", "lamb"]:
        assert getattr(cp1, attr) == getattr(cp2, attr), f"{attr} differs between instances"


def test_invariant_under_round_trip():
    """Compute chi_vev from mu2 and lambda, verify self-consistency."""
    import numpy as np

    from cgc.rp3_engine.chi_potential import ChiPotential

    cp = ChiPotential()

    # chi_vev should satisfy: lambda = -6 * mu2 / (M_P^2 * (chi_vev/M_P)^2)
    chi_from_mu2_lambda = M_P * np.sqrt(-6.0 * cp.mu2 / (M_P**2 * cp.lamb))
    assert_close(cp.chi_vev, chi_from_mu2_lambda, label="chi_vev round-trip")


def test_physical_ranges():
    """All quantities in physically meaningful ranges."""
    from cgc.rp3_engine.chi_potential import ChiPotential

    cp = ChiPotential()

    assert cp.T > 0, f"T={cp.T} non-positive"
    assert 0.0 < cp.alpha < 1.0, f"alpha={cp.alpha} out of (0,1)"
    assert cp.chi_vev > 0, f"chi_vev={cp.chi_vev} non-positive"
    assert cp.chi_vev < 1e19, f"chi_vev={cp.chi_vev} > 10^19 GeV (above Planck)"
    assert cp.lamb > 0, f"lambda={cp.lamb} <= 0 (potential not bounded below)"
    assert cp.mu2 < 0, f"mu2={cp.mu2} >= 0 (no symmetry breaking)"


def run_tests():
    import traceback

    tests = [(name, obj) for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    n_pass = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}")
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL: {name} 鈥?{e}")
        except Exception as e:
            print(f"  ERROR: {name} 鈥?{e}")
            traceback.print_exc()
    print(f"\\n  {n_pass}/{len(tests)} passed")
    return n_pass == len(tests)


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_tests() else 1)
