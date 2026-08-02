"""Phase 2: Run all CGC channels and verify benchmark data."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from cgc import CGCPipeline
from cgc.channels.gauge_field import GaugeFieldStrength, GAUGE_FIELD_BENCHMARK
from cgc.channels.fermion_bilinears import FermionBilinears, FERMION_BENCHMARK
from cgc.channels.higgs_quartic import HiggsQuartic, HIGGS_QUARTIC_BENCHMARK
from cgc.channels.tmunu_spin2 import TMunuSpin2, TMUNU_SPIN2_BENCHMARK

p = CGCPipeline()
channels = []

# Tmu nu (benchmark)
print('='*72)
print('Channel 1: Tμν spin-2 (Phase 1 benchmark)')
print('='*72)
r1 = p.run(TMunuSpin2())
print(r1.summary())
channels.append(('Tμν', r1, TMUNU_SPIN2_BENCHMARK))

# Gauge field strength
print('\n' + '='*72)
print('Channel 2: Fμν^a gauge field strength (Phase 2)')
print('='*72)
r2 = p.run(GaugeFieldStrength('SU(3)'))
print(r2.summary())
channels.append(('Gauge', r2, GAUGE_FIELD_BENCHMARK))

# Fermion bilinear
print('\n' + '='*72)
print('Channel 3: Fermion bilinear ψ̄ψ (Phase 2)')
print('='*72)
r3 = p.run(FermionBilinears('scalar'))
print(r3.summary())
channels.append(('Fermion', r3, FERMION_BENCHMARK))

# Higgs quartic
print('\n' + '='*72)
print('Channel 4: Higgs quartic λ(φ†φ)² (Phase 2)')
print('='*72)
r4 = p.run(HiggsQuartic())
print(r4.summary())
channels.append(('Higgs', r4, HIGGS_QUARTIC_BENCHMARK))

# Summary
print('\n' + '='*72)
print('PHASE 2 CHANNEL SUMMARY')
print('='*72)
print(f'{"Channel":<20} {"Diagrams":<10} {"q=0":<6} {"q≠0":<6} {"Protected":<12} {"Inj≠0":<8} {"Bubbles":<8}')
print('-'*72)
for name, r, bench in channels:
    ds = r.diagram_set
    mc = r.momentum_classification
    tc = r.topology_classification
    cr = r.conservation_report
    rr = r.resummation_result
    print(f'{name:<20} {ds.total_count:<10} {len(mc.zero_transfer):<6} {len(mc.nonzero_transfer):<6} '
          f'{str(cr.verdict.is_protected):<12} {str(rr.injection_nonzero):<8} {len(tc.single_bubble):<8}')

# Verify benchmarks
print('\n' + '='*72)
print('BENCHMARK VERIFICATION')
print('='*72)
for name, r, bench in channels:
    if bench is None:
        print(f'  {name}: no benchmark defined yet')
        continue
    ds = r.diagram_set
    mc = r.momentum_classification
    tc = r.topology_classification
    rr = r.resummation_result
    checks = {
        'total': ds.total_count == bench['expected_total_diagrams'],
        'q0': len(mc.zero_transfer) == bench['expected_q0_count'],
        'q_nonzero': len(mc.nonzero_transfer) == bench['expected_q_nonzero_count'],
        'bubbles': len(tc.single_bubble) == bench['expected_bubble_count'],
        'ladders': len(tc.ladder) == bench['expected_ladder_count'],
    }
    all_ok = all(checks.values())
    print(f'  {name}: {"PASS" if all_ok else "FAIL"}')
    for k, v in checks.items():
        print(f'    {k}: {"OK" if v else "MISMATCH"}')
