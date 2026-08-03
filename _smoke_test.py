"""Import smoke test after RP3 separation (2026-08-03)."""
import sys
sys.path.insert(0, r'D:\论文撰写\Conservation-Guided Classification')

# CGC core (flat-space)
import cgc
from cgc import CGCPipeline
from cgc.engine.pi0_flat_continuum import compute_all
from cgc.engine.conservation_checker import ConservationChecker
from cgc.engine.diagram_generator import DiagramGenerator
from cgc.engine.resummation import Resummator
from cgc.engine.momentum_classifier import MomentumClassifier
from cgc.engine.topology_classifier import TopologyClassifier
print("CGC core (flat-space) imports OK")

# RP3 engine (separate component)
from cgc.rp3_engine.frg_flow_rp3 import RP3Spectrum, RP3TraceDensity
from cgc.rp3_engine.frg_trace_density import FRGTraceDensity
from cgc.rp3_engine.self_consistent_dyson import SelfConsistentSolver
from cgc.rp3_engine.chi_potential import ChiPotential
from cgc.rp3_engine.crossed_ladder_f2 import count_active_modes, compute_crossed_ratio_explicit
print("RP3 engine imports OK")

# Run pi0 flat continuum
results = compute_all()
for r in results:
    print(f"  {r.channel:<35} Pi0={r.pi0:+.4f} sign={r.sign}")
print("ALL IMPORTS + PI0 OK")
