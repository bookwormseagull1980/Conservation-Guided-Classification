#!/usr/bin/env python
"""
CGC Engine Benchmark — Tμν Spin-2 Channel (Phase 2.6)
============================================================================

Runs the complete CGC pipeline on the energy-momentum tensor spin-2
channel and verifies against the known results from Appendix E.

Usage:
    python scripts/benchmark_tmunu.py
    python scripts/benchmark_tmunu.py --output report.json

Verification targets (from Appendix E):
  1. Diagram count matches known enumeration (5 topologically distinct)
  2. Momentum classification: 3 q=0, 2 q≠0
  3. Topology classification: 1 bubble, 2 ladder, 0 crossed/vertex/other
  4. Conservation verdict: protected, Ward identity basis
  5. Injection term nonzero: guaranteed by conservation law
"""

import sys
import os
import io

# Fix Windows console encoding for Unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgc import CGCPipeline
from cgc.channels.tmunu_spin2 import TMunuSpin2, TMUNU_SPIN2_BENCHMARK


def main():
    print("=" * 72)
    print("CGC Engine Benchmark: Tμν Spin-2 Channel")
    print("=" * 72)
    print()

    # ── Initialize ──
    pipeline = CGCPipeline()
    operator = TMunuSpin2()

    print(f"Operator: {operator.name}")
    print(f"  type:          {operator.op_type.name}")
    print(f"  spin channel:  {operator.spin_channel}")
    print(f"  protected:     {operator.is_protected}")
    print(f"  protection:    {operator.protection_source}")
    print()

    # ── Run Pipeline ──
    print("Running CGC pipeline...")
    report = pipeline.run_benchmark(operator, TMUNU_SPIN2_BENCHMARK)

    # ── Print Report ──
    print()
    print(report.summary())

    # ── Benchmark Results ──
    print()
    print("─" * 72)
    print("Benchmark Verification Results")
    print("─" * 72)

    all_passed = True
    for check, passed in report.benchmark_details.items():
        status = "✅" if passed else "❌"
        if not passed:
            all_passed = False
        print(f"  {status} {check}: {passed}")

    print()
    if all_passed:
        print("✅ ALL BENCHMARKS PASSED — CGC engine verified for Tμν spin-2.")
    else:
        print("❌ SOME BENCHMARKS FAILED — see details above.")
        # Note: diagram generation is placeholder; expected to fail until
        # builtin generator is fully implemented (Phase 2.1 milestone)

    # ── Export ──
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
        "benchmark_tmunu_spin2.json",
    )
    report.export_json(output_path)
    print(f"\nReport exported to: {output_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
