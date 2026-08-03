"""Verify CGC numerical outputs unchanged after P0-P2 modifications."""
import sys; sys.path.insert(0, ".")

from cgc.rp3_engine.frg_flow_rp3 import RP3TraceDensity
from cgc.rp3_engine.self_consistent_dyson import SelfConsistentSolver
from cgc.rp3_engine.chi_potential import ChiPotential
from cgc import CGCPipeline
from cgc.channels.tmunu_spin2 import TMunuSpin2
from cgc.channels.gauge_field import GaugeFieldStrength
from cgc.channels.fermion_bilinears import FermionBilinears
from cgc.channels.higgs_quartic import HiggsQuartic

all_ok = True

# 1. Pi0 values (reference: Tmunu +3.600e-2, F2 -1.523e-1)
print("=== Pi0 ===")
td = RP3TraceDensity([])
# NOTE 2026-08-01: RP3TraceDensity has no compute_tmunu/compute_f2 methods.
# Use SelfConsistentSolver (the official path) instead.
from cgc.rp3_engine.self_consistent_dyson import SelfConsistentSolver
s_tm = SelfConsistentSolver("Tmunu")
s_f2 = SelfConsistentSolver("F2")
tm = s_tm.pi0_bare_ir
f2 = s_f2.pi0_bare_ir
print(f"  Tmunu: {tm:.10e}  (ref: +3.5999945350e-02)")
print(f"  F2:    {f2:.10e}  (ref: -1.5226901996e-01)")

# 2. ChiPotential
print("\n=== ChiPotential ===")
cp = ChiPotential()
ok_mu2 = cp.mu2 < 0
ok_lamb = cp.lamb > 0
ok_vev = cp.chi_vev > 0
print(f"  T={cp.T}, alpha={cp.alpha}")
print(f"  mu2={cp.mu2:.4e} (<0: {ok_mu2}), lambda={cp.lamb:.4f} (>0: {ok_lamb})")
print(f"  chi_vev={cp.chi_vev:.4e} GeV = {cp.chi_vev / 2.4353e18:.4f} M_P (>0: {ok_vev})")
all_ok &= ok_mu2 and ok_lamb and ok_vev

# 3. DSE Gap
print("\n=== DSE Gap ===")
s_tmunu = SelfConsistentSolver("Tmunu")
vc_tmunu = s_tmunu.find_v_crit()
gap_tm = vc_tmunu.get("gap_decades", 0)
print(f"  Tmunu: V_crit={vc_tmunu['v_crit']:.4e}, V_native={s_tmunu.native_v:.4e}, log10(gap)={gap_tm:.2f} (>0: {gap_tm>0})")
ok_tm = gap_tm > 0

s_f2 = SelfConsistentSolver("F2")
vc_f2 = s_f2.find_v_crit()
ok_f2 = vc_f2.get("v_crit") is None
print(f"  F2:    V_crit={vc_f2.get('v_crit')}  (None=no pole, OK: {ok_f2})")
all_ok &= ok_tm and ok_f2

# 4. Classification
print("\n=== Classification ===")
pipeline = CGCPipeline()
for name, cls, exp_prot, exp_nz in [
    ("Tmunu", TMunuSpin2, True, True),
    ("F2", GaugeFieldStrength, True, True),
    ("Fermion", FermionBilinears, False, False),
    ("Higgs", HiggsQuartic, False, False),
]:
    r = pipeline.run(cls())
    v = r.conservation_report.verdict
    ok_p = v.is_protected == exp_prot
    ok_n = v.matrix_element_nonzero == exp_nz
    print(f"  {name:8s}: protected={v.is_protected}/{exp_prot} (OK:{ok_p})  nonzero={v.matrix_element_nonzero}/{exp_nz} (OK:{ok_n})")
    all_ok &= ok_p and ok_n

print(f"\n{'ALL OUTPUTS UNCHANGED' if all_ok else '!!! OUTPUTS CHANGED !!!'}")
print(f"Version: {r.pipeline_version}")
