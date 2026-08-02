# cgc/channels — Physical channel implementations
# Each channel defines: operator, protected/unprotected status, external legs, relevant vertices

from .fermion_bilinears import FermionBilinears
from .gauge_field import GaugeFieldStrength
from .higgs_quartic import HiggsQuartic
from .tmunu_spin2 import TMunuSpin2

__all__ = [
    "TMunuSpin2",
    "GaugeFieldStrength",
    "FermionBilinears",
    "HiggsQuartic",
]
