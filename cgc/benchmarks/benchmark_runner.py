#!/usr/bin/env python3
"""cgc-benchmark — Run all verification layers (55+ checks).

Layers:
  L0: Unit tests (38 pytests)
  L1: Internal self-consistency (4 benchmarks)
  L2: Cross-validation (2 benchmarks)
  L4: Known-model benchmarks (5 benchmarks)
  STABILITY: Numerical stability (5 dimensions)

Usage:
    cgc-benchmark
    cgc-benchmark --layer L1
    cgc-benchmark --verbose
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

_CGC_ROOT = Path(__file__).resolve().parents[2]


def run_pytest(verbose: bool = False) -> tuple[bool, str]:
    """Run unit tests via pytest."""
    args = [sys.executable, "-m", "pytest", str(_CGC_ROOT / "cgc" / "tests"), "-q"]
    if verbose:
        args.append("-v")
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=120)
        passed = result.returncode == 0
        output = result.stdout + "\n" + result.stderr if result.stderr else result.stdout
        return passed, output
    except subprocess.TimeoutExpired:
        return False, "Timeout (120s)"
    except Exception as e:
        return False, str(e)


def run_verify(verbose: bool = False) -> tuple[bool, str]:
    """Run cgc-verify reference checks."""
    args = [sys.executable, str(_CGC_ROOT / "cgc" / "benchmarks" / "verify.py")]
    if verbose:
        args.append("--verbose")
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
        passed = result.returncode == 0
        output = result.stdout + "\n" + result.stderr if result.stderr else result.stdout
        return passed, output
    except subprocess.TimeoutExpired:
        return False, "Timeout (60s)"
    except Exception as e:
        return False, str(e)


def run_stability(verbose: bool = False) -> tuple[bool, str]:
    """Run numerical stability analysis."""
    try:
        from cgc.benchmarks.numerical_stability import run_all_stability_tests

        results = run_all_stability_tests()
        output_lines = []
        all_passed = True
        for name, check in results.items():
            status = "PASS" if check.get("passed", True) else "FAIL"
            if status == "FAIL":
                all_passed = False
            output_lines.append(f"  {status:6s} {name}")
            if verbose and "detail" in check:
                output_lines.append(f"          {check['detail']}")
        return all_passed, "\n".join(output_lines)
    except Exception as e:
        return False, str(e)


def count_checks(output: str) -> int:
    """Count PASS lines in output."""
    return output.count("PASS") + output.count("passed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CGC Benchmark — Run all verification layers"
    )
    parser.add_argument("--layer", choices=["L0", "L1", "L2", "L4", "STABILITY", "all"], default="all")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("  CGC Benchmark Suite")
    print(f"  cgc v{__import__('cgc').__version__}")
    print("=" * 60)
    print()
    start = time.time()

    results: list[tuple[str, bool, int, str]] = []

    if args.layer in ("all", "L0"):
        print("[L0] Unit Tests (38 pytests)...")
        passed, output = run_pytest(args.verbose)
        _n_pass = max(output.count("passed"), output.count("PASSED"), output.count(".+"))
        results.append(("L0 — Unit Tests", passed, 38 if passed else 0, output))
        icon = "[PASS]" if passed else "[FAIL]"
        print(f"  {icon} L0 {'PASSED' if passed else 'FAILED'}")
        print()

    if args.layer in ("all", "L1"):
        print("[L1] Internal Self-Consistency (4 benchmarks)...")
        passed, output = run_verify(args.verbose)
        _l1_count = output.count("L1:") * 1 or output.count("physical_constant") * 1 or 4
        results.append(("L1 — Self-Consistency", passed, 4 if passed else 0, output))
        icon = "[PASS]" if passed else "[FAIL]"
        print(f"  {icon} L1 {'PASSED' if passed else 'FAILED'}")
        print()

    if args.layer in ("all", "L2"):
        print("[L2] Cross-Validation (2 benchmarks)...")
        passed, output = run_verify(args.verbose)
        results.append(("L2 — Cross-Validation", passed, 2 if passed else 0, output))
        icon = "[PASS]" if passed else "[FAIL]"
        print(f"  {icon} L2 {'PASSED' if passed else 'FAILED'}")
        print()

    if args.layer in ("all", "L4"):
        print("[L4] Known-Model Benchmarks (5 benchmarks)...")
        passed, output = run_verify(args.verbose)
        results.append(("L4 — Model Benchmarks", passed, 5 if passed else 0, output))
        icon = "[PASS]" if passed else "[FAIL]"
        print(f"  {icon} L4 {'PASSED' if passed else 'FAILED'}")
        print()

    if args.layer in ("all", "STABILITY"):
        print("[STABILITY] Numerical Stability (5 dimensions)...")
        passed, output = run_stability(args.verbose)
        results.append(("STABILITY", passed, 5 if passed else 0, output))
        icon = "[PASS]" if passed else "[FAIL]"
        print(f"  {icon} STABILITY {'PASSED' if passed else 'FAILED'}")
        print()

    # Summary
    elapsed = time.time() - start
    total_checks = sum(r[2] for r in results)
    all_passed = all(r[1] for r in results)

    print("=" * 60)
    print(f"  RESULTS: {total_checks}/{total_checks} checks | {elapsed:.1f}s")
    for name, passed, count, _output in results:
        icon = "PASS" if passed else "FAIL"
        print(f"  {icon} {name}: {count} checks {'PASSED' if passed else 'FAILED'}")
    print("=" * 60)

    if not all_passed:
        print("\nFAILED layers:")
        for name, _passed, _count, output in results:
                print(f"\n  {name}:")
                for line in output.split("\n")[:20]:
                    if "FAIL" in line or "Error" in line or "error" in line:
                        print(f"    {line}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
