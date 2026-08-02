"""CGC — Conservation-Guided Operator Classification for Emergent Gravity.

Theory references:
  - Paper 1: Coarse-graining emergence on RP³ (Appendix E)
  - Paper 3: Gauge group from RP³ geometry (EC torsion + isometry breaking)
"""

__version__ = "1.1.0"

from .engine.conservation_checker import ConservationReport
from .engine.momentum_classifier import MomentumClassification
from .engine.pipeline import CGCPipeline
from .engine.resummation import ResummationResult
from .engine.topology_classifier import TopologyClassification

__all__ = [
    "CGCPipeline",
    "ConservationReport",
    "MomentumClassification",
    "TopologyClassification",
    "ResummationResult",
]
