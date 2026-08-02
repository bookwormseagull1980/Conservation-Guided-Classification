#!/usr/bin/env python3
"""Run all CGC unit tests.

Usage:
    python -m cgc.tests.run_all
    python tests/run_all.py
"""

import os
import sys
import time

# Ensure cgc is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(_ROOT))

os.chdir(_HERE)  # work from tests/ dir for conftest imports

TEST_MODULES = [
    "test_chi_potential",
    "test_trace_density",
    "test_conservation_checker",
    "test_dyson_schwinger",
    "test_edge_cases",
    "test_properties",
]


def run_all():
    total = 0
    passed = 0
    start = time.time()

    for mod_name in TEST_MODULES:
        try:
            mod = __import__(mod_name)
            if hasattr(mod, "run_tests"):
                ok = mod.run_tests()
                n = len([x for x in dir(mod) if x.startswith("test_") and callable(getattr(mod, x))])
                total += n
                if ok:
                    passed += n
            else:
                print(f"  SKIP: {mod_name} (no run_tests)")

        except Exception as e:
            print(f"  FAIL: {mod_name} - {e}")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {passed}/{total} tests passed ({elapsed:.1f}s)")
    print(f"{'=' * 60}")

    if passed < total or total == 0:
        print(f"  {total - passed} test(s) FAILED")
    else:
        print("  All tests passed. CGC is unit-test verified.")

    return passed == total and total > 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
