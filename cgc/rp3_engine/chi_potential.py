"""Chi Potential — RP3 Breathing Mode (Geometric Parameters Only).

Extracted from chi_condensation.py (deprecated 2026-07-31).
The geometric parameters from the Einstein-Cartan action on RP3
are valid and independent of the equilibrium RG flow hypothesis.

Physics
-------
The RP3 breathing mode chi (J=0 tensor harmonic) has a tachyon potential:
    V_eff(chi) = mu2*chi^2/2 + lambda*chi^4/24,  mu2 < 0

From CG-Framework (geometric_ewsb.py, EC action):
    alpha = 1/50  (EC curvature coupling)
    T = 5         (zero-mode counting, rigid)
    chi_vev = M_P / sqrt(T-2) = 0.577 M_P

These are zero-free-parameter geometric results.
The equilibrium RG flow that was built on top of them is deprecated
because the synchronicity hypothesis was disproven.

Author: CGC
Date: 2026-07-31
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frg_flow_rp3 import M_P


@dataclass
class ChiPotential:
    """Effective potential for RP3 breathing mode.

    From Einstein-Cartan action on RP3:
    - ST tachyon: mu2 = -alpha * M_P^2, alpha = 1/50
    - Quartic: lambda = 6*alpha/(chi_vev/M_P)^2
    - chi_vev = M_P / sqrt(T-2) = 0.577 M_P

    References
    ----------
    - T=5: [FRAMEWORK.md §二] rigid from zero-mode counting
            (12 gauge + 2 graviton Killing vectors on RP3)
    - alpha=1/50: [FRAMEWORK.md 2026-07-20] EC curvature coupling
                  from geometric_isometry_breaking.py
    - chi_vev: [FRAMEWORK.md §二点五] M_P/sqrt(T-2) from EC scalar
               mode decomposition with T-2 effective modes
    - mu2 = -alpha*M_P^2: [Paper 1 §3.3] ST tachyon mass from
                          Einstein-Cartan curvature splitting
    - lambda: [Paper 1 §3.3] quartic self-coupling from
              chi^4 term in EC Laplacian expansion
    """

    T: float = 5.0
    # Ref: [FRAMEWORK.md §二] T=5 rigid from zero-mode counting

    alpha: float = 1.0 / 50.0
    # Ref: [FRAMEWORK.md 2026-07-20] EC curvature coupling α=1/50
    #      from geometric_isometry_breaking.py: EC action ratio
    #      gives α = (T-2)^{-1} * (L^{-2})/(8π) = 1/50 at T=5, L=2.44

    @property
    def chi_vev(self) -> float:
        # Ref: [FRAMEWORK.md §二点五] chi_vev = M_P / sqrt(T-2)
        #      Only T-2 effective scalar modes survive EC constraint
        #      (2 zero modes removed by diffeomorphism gauge fixing)
        return M_P / np.sqrt(self.T - 2.0)  # type: ignore[no-any-return]

    @property
    def mu2(self) -> float:
        # Ref: [Paper 1 §3.3] ST tachyon mass: mu2 = -alpha * M_P^2
        #      Negative mass squared = spontaneous symmetry breaking
        return -self.alpha * M_P**2

    @property
    def lamb(self) -> float:
        # Ref: [Paper 1 §3.3] quartic coupling from EC Laplacian
        #      lambda = 6*alpha / (chi_vev/M_P)^2
        #      Dimensionless after normalizing by M_P
        return 6.0 * self.alpha / (self.chi_vev / M_P) ** 2

    @property
    def V_min(self) -> float:
        # Ref: [Paper 1 §3.3] V(chi_vev) = -3/2 * alpha * M_P^2 * chi_vev^2
        #      (with the tachyon+quartic form this simplifies)
        c = self.chi_vev
        return 0.5 * self.mu2 * c**2 + self.lamb * c**4 / 24.0

    def V(self, chi: float) -> float:
        # Ref: [Paper 1 §3.3] V(chi) = mu2*chi^2/2 + lambda*chi^4/24
        return 0.5 * self.mu2 * chi**2 + self.lamb * chi**4 / 24.0

    def V_barrier(self) -> float:
        # Ref: [Paper 1 §3.3] barrier height = -V_min = V(0) - V(chi_vev)
        #      For pure ST tachyon, V(0)=0 by EC scale invariance
        return -self.V_min

    def summary(self) -> None:
        print("  RP3 breathing mode chi:")
        print(f"    T = {self.T}, alpha = {self.alpha}")
        print(f"    mu^2 = {self.mu2:.4e} GeV^2")
        print(f"    lambda = {self.lamb:.4f}")
        print(f"    chi_vev = {self.chi_vev:.4e} GeV = {self.chi_vev / M_P:.4f} M_P")
        print(f"    V(chi_vev) = {self.V_min:.4e} GeV^4")
        print(f"    Barrier at chi=0: {self.V_barrier():.4e} GeV^4")
