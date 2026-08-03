#!/usr/bin/env python3
"""Generate reference_output.json from live CGC computation (v2.0).

Covers: L1 physical constants, channel Pi0, Dyson-Schwinger,
L2 Pi0 cross-validation, CG-Framework references, L4 model benchmarks.
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

_CGC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _CGC_ROOT not in sys.path:
    sys.path.insert(0, _CGC_ROOT)


def gather_physical_constants() -> dict:
    from cgc.rp3_engine.chi_potential import ChiPotential
    from cgc.rp3_engine.frg_flow_rp3 import L_RP3, M_CURV, M_P

    cp = ChiPotential()
    V_RP3 = np.pi**2 * (L_RP3 / M_P) ** 3
    K_POLE = np.sqrt(abs(cp.mu2))

    return {
        "M_P_GeV": float(M_P),
        "M_CURV_GeV": float(M_CURV),
        "L_RP3": float(L_RP3),
        "V_RP3_GeVm3": float(V_RP3),
        "K_POLE_GeV": float(K_POLE),
        "K_POLE_over_MP": float(K_POLE / M_P),
        "T_flavor": cp.T,
        "alpha": cp.alpha,
        "lambda_chi_uv": cp.lamb,
        "mu2_GeV2": float(cp.mu2),
        "chi_vev_uv_GeV": float(cp.chi_vev),
        "chi_vev_over_MP": float(cp.chi_vev / M_P),
    }


def gather_channels() -> dict:
    from cgc.engine.diagram_generator import OperatorSpec, OperatorType
    from cgc.engine.one_loop_generator import expected_one_loop_count
    from cgc.rp3_engine.self_consistent_dyson import SelfConsistentSolver

    channels = {}
    for name, op_type, lorentz_rank, spin, ext_mom, mass_dim, is_prot in [
        ("Tmunu", OperatorType.CONSERVED_CURRENT, 2, 2, 2, 4, True),
        ("F2", OperatorType.GAUGE_FIELD_STRENGTH, 2, 2, 2, 4, True),
        ("FermionBilinear", OperatorType.UNPROTECTED_FERMION, 0, 0, 2, 3, False),
        ("HiggsQuartic", OperatorType.UNPROTECTED_SCALAR, 0, 0, 2, 4, False),
    ]:
        s = SelfConsistentSolver(name)
        spec = OperatorSpec(
            name=name,
            op_type=op_type,
            lorentz_rank=lorentz_rank,
            spin_channel=spin,
            external_momenta=ext_mom,
            mass_dimension=mass_dim,
            is_protected=is_prot,
            protection_source="",
        )
        n_diag = expected_one_loop_count(spec)
        pi0 = s.pi0_bare_ir
        verdict = "DYNAMIC_EMERGENCE" if pi0 > 0 else "TOPOLOGICAL_EMERGENCE"
        if not is_prot:
            verdict = "NO_EMERGENCE"

        channels[name] = {
            "pi0_bare_ir": float(pi0),
            "n_diagrams_total": n_diag,
            "emergence_verdict": verdict,
            "is_protected": is_prot,
            "V_native": float(s.native_v),
        }

    return channels


def gather_dyson_schwinger() -> dict:
    from cgc.rp3_engine.dyson_schwinger import DysonSchwingerSolver

    result = {}
    for ch in ["Tmunu", "F2"]:
        dse = DysonSchwingerSolver(ch)
        res = dse.scan_V()
        result[ch] = {}
        for key_src, key_dst in [
            ("Pi0_bare", "pi0_bare"),
            ("Pi0_bubble", "pi0_bubble"),
            ("V_native", "v_native"),
            ("V_crit_tadpole", "v_crit_tadpole"),
            ("V_crit_bubble_bare", "v_crit_bubble_bare"),
        ]:
            v = res.summary.get(key_src)
            result[ch][key_dst] = float(v) if v is not None and abs(float(v)) != float("inf") else "inf"

    return result


def gather_pi0_cross_validation() -> dict:
    """L2: Pi0 computed on two INDEPENDENT code paths."""
    import numpy as np

    from cgc.rp3_engine.frg_flow_rp3 import (
        M_P,
        LitimRegulator,
        RP3TraceDensity,
        f2_field_content,
        tmunu_field_content,
    )
    from cgc.rp3_engine.self_consistent_dyson import SelfConsistentSolver

    result = {}
    for name, fields_fn in [("Tmunu", tmunu_field_content), ("F2", f2_field_content)]:
        s = SelfConsistentSolver(name)
        pi0_cgc = s.pi0_bare_ir

        fields = fields_fn()
        trace = RP3TraceDensity(fields, regulator=LitimRegulator())
        k_grid = np.geomspace(1.0, M_P, 500)
        d_ln = np.log(k_grid[1] / k_grid[0])
        eta = np.array([trace.trace_density_at_k(k) for k in k_grid])
        pi0_frg = np.cumsum(eta[::-1])[::-1][0] * d_ln

        result[name] = {
            "pi0_bare_ir": float(pi0_cgc),
            "pi0_frg_path": float(pi0_frg),
            "paths_identical": bool(abs(pi0_cgc - pi0_frg) < 1e-15),
        }

    return result


def gather_CG_Framework_refs() -> dict:
    """L2: CG-Framework hard reference values (from independent codebase cg_frg/)."""
    return {
        "Z_phys_M_G": 0.99805,
        "G_N_deviation_pct": 0.027,
        "gamma_M": 0.0,
        "gamma_M_precision": 1e-16,
        "TT_pole_exists": True,
        "TT_propagator": "G_TT ~ k^-2 (delta-criterion: True)",
        "matter_backreaction_pct": 0.195,
        "source_files": [
            "cg_frg/tt_tensor.py",
            "cg_frg/newton.py",
            "cg_frg/sigma_k_definitive.py",
        ],
    }


def gather_chi_flow() -> dict:
    return {
        "status": "DEPRECATED",
        "reason": "Equilibrium RG flow hypothesis disproven",
        "deprecated_date": "2026-07-31",
        "replaced_by": "CG-Framework cg_frg/tt_tensor.py (Z_phys = 0.998)",
    }


def main():
    data = {
        "schema_version": "1.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "deprecation_note": (
            "Equilibrium RG flow modules (chi_condensation, chi_effective_potential, "
            "coupled_chi_*) deprecated 2026-07-31. Synchronicity hypothesis disproven. "
            "L2 Pi0 cross-validation and CG-Framework references added as replacements."
        ),
        "physical_constants": gather_physical_constants(),
        "channels": gather_channels(),
        "chi_effective_potential": gather_chi_flow(),
        "dyson_schwinger": gather_dyson_schwinger(),
        "pi0_cross_validation": gather_pi0_cross_validation(),
        "cg_framework_references": gather_CG_Framework_refs(),
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"reference_output.json written ({out_path})")


if __name__ == "__main__":
    main()
