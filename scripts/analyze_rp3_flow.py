import sys, io, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
from cgc.rp3_engine.frg_flow_rp3 import *

spectrum = RP3Spectrum(L_RP3)
print('=== MODE COUNTING ON RP3 (L=%.2f, M_CURV=%.2e GeV) ===' % (L_RP3, M_CURV))
print()
for k_label, k_val, k_ratio in [
    ('k=10*M_CURV', 10*M_CURV, 10), ('k=M_P', M_P, M_P/M_CURV),
    ('k=M_G', M_G, M_G/M_CURV), ('k=M_CURV', M_CURV, 1.0),
    ('k=M_CURV/2', M_CURV/2, 0.5), ('k=M_CURV/10', M_CURV/10, 0.1),
    ('k=1 TeV', 1e3, 1e3/M_CURV),
]:
    n_vec = spectrum.count_modes_below(k_val, FieldSpecies.VECTOR)
    n_sp = spectrum.count_modes_below(k_val, FieldSpecies.SPINOR)
    n_sc = spectrum.count_modes_below(k_val, FieldSpecies.SCALAR)
    print(f'{k_label:>16s} (k/M_CURV={k_ratio:.2f}): '
          f'vector={n_vec:>5d}  spinor={n_sp:>5d}  scalar={n_sc:>5d}')

print()
print('=== FIRST EIGENVALUES ===')
print(f'Vector  n=1: lambda = 4 * M_CURV^2,  sqrt = {2*M_CURV:.4e} GeV')
print(f'Spinor  n=0: lambda = 2.25 * M_CURV^2, sqrt = {1.5*M_CURV:.4e} GeV')
print(f'Scalar  J=0: lambda = 0 (ZERO MODE)')
print(f'Scalar  J=2: lambda = 8 * M_CURV^2 = {8*M_CURV**2:.4e} GeV^2')
print(f'M_CURV = {M_CURV:.4e} GeV')
print(f'M_P    = {M_P:.4e} GeV')
print(f'M_G    = {M_G:.4e} GeV (ratio M_G/M_CURV = {M_G/M_CURV:.3f})')
print(f'log(M_P/M_CURV) = {np.log(M_P/M_CURV):.3f} e-folds of UV flow')

print()
print('=== BETA SIGN ANALYSIS FOR F2 ===')
solver = RP3FRGFlowSolver(FlowConfig(operator_name='F2'))
for k_label, k_val in [
    ('k=5*M_CURV', 5*M_CURV), ('k=3*M_CURV', 3*M_CURV),
    ('k=2*M_CURV', 2*M_CURV), ('k=1.5*M_CURV', 1.5*M_CURV),
    ('k=M_CURV', M_CURV), ('k=0.5*M_CURV', 0.5*M_CURV),
]:
    I = solver.compute_I(k_val)
    sign = '+' if I > 0 else ('-' if I < 0 else '0')
    dv_per_efold = I * 1.56e-3**2 * 0.1
    print(f'{k_label:>16s}: I = {I:+.6e}  sign = {sign}  dV/e-fold ~ {dv_per_efold:+.2e}')

print()
print('=== BETA SIGN ANALYSIS FOR Tmunu ===')
solver2 = RP3FRGFlowSolver(FlowConfig(operator_name='Tmunu'))
for k_label, k_val in [
    ('k=5*M_CURV', 5*M_CURV), ('k=3*M_CURV', 3*M_CURV),
    ('k=2*M_CURV', 2*M_CURV), ('k=1.5*M_CURV', 1.5*M_CURV),
    ('k=M_CURV', M_CURV), ('k=0.5*M_CURV', 0.5*M_CURV),
    ('k=1 TeV', 1e3), ('k=1 GeV', 1.0),
]:
    I = solver2.compute_I(k_val)
    sign = '+' if I > 0 else ('-' if I < 0 else '0')
    dv_per_efold = I * 1.79e-4**2 * 0.1
    print(f'{k_label:>16s}: I = {I:+.6e}  sign = {sign}  dV/e-fold ~ {dv_per_efold:+.2e}')

print()
print('=== PHYSICS SUMMARY ===')
print(f'1. At k > {2*M_CURV:.2e} GeV (k > 2*M_CURV): vector modes exist, I < 0 (fermion-dominated)')
print(f'2. At {1.5*M_CURV:.2e} < k < {2*M_CURV:.2e} GeV: NO vector modes, only spinor modes')
print(f'3. At k < {1.5*M_CURV:.2e} GeV (k < 1.5*M_CURV): NO vector OR spinor modes — I = 0 for F2')
print(f'4. Scalars have zero mode — Tmunu always has non-zero I, but very small at low k')
print(f'5. Total RG flow range above M_CURV: only {np.log(M_P/M_CURV):.2f} e-folds')
print(f'6. V flows only above M_CURV, FREEZES below M_CURV (modes run out)')
print(f'7. dV_total ~ V^2 * I * ln(M_P/M_CURV) ~ 1e-7 << V ~ 1e-3')
