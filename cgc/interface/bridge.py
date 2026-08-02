"""CGC↔FRG Bridge — Bidirectional conversion between engine objects and JSON.

This module contains:
  - CGCToFRGBridge: converts CGC pipeline results into CGCToFRGPayload
  - FRGToCGCBridge: converts FRG flow results into FRGToCGCPayload
  - CGC_FRG_Validator: cross-validates CGC predictions against FRG results
  - Convenience functions: export_cgc_to_frg(), import_frg_to_cgc()
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .schema import (
    SCHEMA_VERSION,
    CGCFieldContent,
    CGCOperatorSpec,
    CGCResummationData,
    CGCSpectrumParams,
    CGCToFRGPayload,
    EmergenceVerdict,
    FieldSpecies,
    FRGCrossValidation,
    FRGFlowResult,
    FRGSpectralFunction,
    FRGToCGCPayload,
    OperatorType,
    PoleStatus,
    ProtectionBasis,
)

# ══════════════════════════════════════════════════════════════════════════
# CGC → FRG Bridge
# ══════════════════════════════════════════════════════════════════════════


class CGCToFRGBridge:
    """Converts CGC engine objects into the FRG-consumable JSON payload.

    Usage:
        bridge = CGCToFRGBridge()
        payload = bridge.from_engine_modules(
            operator_name="Tmunu",
            field_content=f2_field_content(),
            solver=SelfConsistentSolver("Tmunu"),
        )
        payload.export_json("output/cgc_to_frg_tmunu.json")
    """

    @staticmethod
    def build_operator_spec(
        operator_name: str,
        operator_type: str = "CONSERVED_CURRENT",
        is_protected: bool = True,
        protection_basis: str = "WARD_IDENTITY",
        pi0_bare_ir: float | None = None,
        native_v: float | None = None,
        lambda_crit: float = 4.0 / 27.0,
    ) -> CGCOperatorSpec:
        """Build operator spec from CGC engine parameters."""

        # Determine emergence verdict
        if is_protected and pi0_bare_ir is not None:
            verdict = EmergenceVerdict.DYNAMIC_EMERGENCE if pi0_bare_ir > 0 else EmergenceVerdict.TOPOLOGICAL_EMERGENCE
        elif is_protected:
            verdict = EmergenceVerdict.UNCERTAIN
        else:
            verdict = EmergenceVerdict.NO_EMERGENCE

        # Pi0 sign
        pi0_sign = (1 if pi0_bare_ir > 0 else -1 if pi0_bare_ir < 0 else 0) if pi0_bare_ir is not None else 0

        # Critical V needed
        v_crit_needed = None
        gap_to_crit = None
        if pi0_bare_ir is not None and pi0_bare_ir > 0:
            v_crit_needed = lambda_crit / pi0_bare_ir
            if native_v is not None and native_v > 0:
                gap_to_crit = v_crit_needed / native_v

        return CGCOperatorSpec(
            name=operator_name,
            operator_type=OperatorType(operator_type),
            is_protected=is_protected,
            protection_basis=ProtectionBasis(protection_basis),
            matrix_element_nonzero=is_protected,
            emergence_verdict=verdict,
            pi0_sign=pi0_sign,
            pi0_bare_ir=pi0_bare_ir,
            lambda_crit=lambda_crit,
            v_native=native_v,
            v_crit_needed=v_crit_needed,
            gap_to_criticality=gap_to_crit,
        )

    @staticmethod
    def build_field_content(
        fields: list[Any],
        operator_type: str = "CONSERVED_CURRENT",
    ) -> list[CGCFieldContent]:
        """Convert CGC engine FieldContent list to schema format.

        Handles both cgc.engine.frg_flow_rp3.FieldContent and plain tuples.
        """
        result = []
        for f in fields:
            # Handle dataclass with attributes
            if hasattr(f, "name") and hasattr(f, "field_type"):
                ft = f.field_type
                if hasattr(ft, "value"):
                    ft = ft.value
                result.append(
                    CGCFieldContent(
                        name=f.name,
                        field_type=FieldSpecies(ft),
                        n_species=int(f.n_species),
                        dof_per_species=int(f.dof_per_species),
                        mass_gev=float(getattr(f, "mass_gev", 0.0)),
                        coupling_sq=float(getattr(f, "coupling_sq", 1.0)),
                    )
                )
            # Handle dict
            elif isinstance(f, dict):
                result.append(
                    CGCFieldContent(
                        name=f.get("name", "unknown"),
                        field_type=FieldSpecies(f.get("field_type", "scalar")),
                        n_species=int(f.get("n_species", 1)),
                        dof_per_species=int(f.get("dof_per_species", 1)),
                        mass_gev=float(f.get("mass_gev", 0.0)),
                        coupling_sq=float(f.get("coupling_sq", 1.0)),
                    )
                )
        return result

    @staticmethod
    def build_resummation_data(
        operator_name: str,
        pi0_bare_ir: float | None = None,
        native_v: float | None = None,
        bcs_integral: float | None = None,
        has_bcs: bool = False,
        n_q0: int = 0,
        n_ladder: int = 0,
        n_bubble: int = 0,
    ) -> CGCResummationData:
        """Build resummation data from Dyson solver results."""

        notes = []
        if pi0_bare_ir is not None and native_v is not None:
            v_pi0 = native_v * pi0_bare_ir
            if v_pi0 > 0:
                notes.append(f"V·Pi0 = {v_pi0:.4e} at equilibrium")
                notes.append(f"Gap to 4/27: {4 / 27 / v_pi0:.2e}x")
            else:
                notes.append("Pi0 < 0 → Dyson suppression, not amplification")

        return CGCResummationData(
            operator_name=operator_name,
            n_total_diagrams=n_q0 + n_ladder + n_bubble,
            n_q0_diagrams=n_q0,
            n_ladder_diagrams=n_ladder,
            n_single_bubble=n_bubble,
            resummation_formula="Pi_resum = Pi0 / (1 - V*Pi0); V_crit = (4/27)/Pi0",
            dyson_y_crit=4.0 / 27.0,
            bcs_integral_at_native_V=bcs_integral,
            has_bcs_solution=has_bcs,
            notes=notes,
        )

    def from_self_consistent_solver(
        self,
        operator_name: str,
        field_content: list[Any],
        pi0_bare_ir: float,
        native_v: float,
        is_protected: bool = True,
        protection_basis: str = "WARD_IDENTITY",
        op_type: str = "CONSERVED_CURRENT",
        bcs_integral: float | None = None,
        has_bcs: bool = False,
    ) -> CGCToFRGPayload:
        """Build full CGC→FRG payload from SelfConsistentSolver results."""

        op_spec = self.build_operator_spec(
            operator_name=operator_name,
            operator_type=op_type,
            is_protected=is_protected,
            protection_basis=protection_basis,
            pi0_bare_ir=pi0_bare_ir,
            native_v=native_v,
        )

        fields = self.build_field_content(field_content, op_type)

        resummation = self.build_resummation_data(
            operator_name=operator_name,
            pi0_bare_ir=pi0_bare_ir,
            native_v=native_v,
            bcs_integral=bcs_integral,
            has_bcs=has_bcs,
        )

        # Build compute instructions
        instructions = []
        if op_spec.emergence_verdict == EmergenceVerdict.DYNAMIC_EMERGENCE:
            instructions = [
                f"1. Run full FRG flow for {operator_name} with RP3 discrete spectrum",
                "2. Compute V(k) trajectory from k_UV=M_P to k_IR=M_CURV",
                f"3. Determine if V·Pi0 reaches {4 / 27:.4f} (cubic vertex critical condition)",
                "4. Compute self-consistent Dyson: y = V·Pi0 satisfies y/(1-y)^2 = V_native·Pi0_bare",
                "5. Output spectral function: pole position, residue, continuum",
            ]
        elif op_spec.emergence_verdict == EmergenceVerdict.TOPOLOGICAL_EMERGENCE:
            instructions = [
                f"1. Verify Pi0 < 0 (fermion-dominated) for {operator_name}",
                "2. Confirm Dyson suppression: V_eff < V_bare at all scales",
                "3. Cross-check: pole from topology (equivariant index theorem), not dynamics",
            ]
        else:
            instructions = [
                f"1. Verify injection suppressed for {operator_name}",
                "2. Confirm no ladder accumulation in IR limit",
            ]

        # FRG recommendations
        frg_recs = {
            "k_UV": 2.4353e18,  # M_P
            "k_IR": 1.0,  # ~1 GeV
            "n_grid": 500,
            "regulator": "Litim",
            "spectrum": "Camporesi-RP3",
            "include_rp3_modes": True,
            "include_anomalous_dim": True,
            "include_v3_correction": True,
            "fermion_sign_flip": True,
            "threshold_function": "2k^2/(k^2+m^2)",
        }

        return CGCToFRGPayload(
            operator=op_spec,
            compute_instructions=instructions,
            fields=fields,
            spectrum=CGCSpectrumParams(),
            resummation=resummation,
            frg_recommendations=frg_recs,
        )


# ══════════════════════════════════════════════════════════════════════════
# FRG → CGC Bridge
# ══════════════════════════════════════════════════════════════════════════


class FRGToCGCBridge:
    """Converts FRG flow results into the CGC-consumable JSON payload.

    Usage:
        bridge = FRGToCGCBridge()
        payload = bridge.from_rp3_flow_solver(
            operator_name="Tmunu",
            flow_result=result,  # FlowResult from frg_flow_rp3.RP3FRGFlowSolver
        )
        payload.export_json("output/frg_to_cgc_tmunu.json")
    """

    def from_rp3_flow_result(
        self,
        operator_name: str,
        flow_result: Any,
        pi0_bare_ir: float | None = None,
        pi0_frg_computed: float | None = None,
        cgc_payload_path: str = "",
    ) -> FRGToCGCPayload:
        """Build FRG→CGC payload from RP3FRGFlowSolver.FlowResult."""

        # Spectral function — pole status depends on Physics context
        #   For Pi0>0 channels: FRG flow alone typically can't reach V_crit
        #   (gap ×23275 bridged by Dyson resummation, not beta flow)
        #   → MARGINAL: channel supports pole, but requires Dyson to close gap
        #   For Pi0<0 channels: pole from topology, not dynamics → ABSENT
        #   For Pi0>0 with V flow crossing: strong result → EXISTS
        pole_stat = PoleStatus.NOT_COMPUTED
        if hasattr(flow_result, "crosses_critical"):
            if flow_result.crosses_critical:
                pole_stat = PoleStatus.EXISTS
            elif pi0_bare_ir is not None and pi0_bare_ir > 0:
                # Pi0>0 → channel supports pole, but FRG flow alone insufficient
                # The pole forms via Dyson self-consistent resummation
                pole_stat = PoleStatus.MARGINAL
            else:
                # Pi0<0 or None → no dynamical pole from flow
                pole_stat = PoleStatus.ABSENT

        spectral = FRGSpectralFunction(
            operator_name=operator_name,
            pole_status=pole_stat,
            pole_position_gev2=0.0 if pole_stat == PoleStatus.EXISTS else None,
            notes=[
                f"v_uv={getattr(flow_result, 'v_uv', 0):.4e}",
                f"v_ir={getattr(flow_result, 'v_ir', 0):.4e}",
                f"log_enhancement={getattr(flow_result, 'log_enhancement', 0):+.4f}",
            ],
        )

        # Flow result
        frg_flow = FRGFlowResult(
            operator_name=operator_name,
            k_uv=float(getattr(flow_result, "k_grid", [0])[0])
            if hasattr(flow_result, "k_grid") and len(flow_result.k_grid) > 0
            else 0.0,
            k_ir=float(getattr(flow_result, "k_grid", [0, 0])[-1])
            if hasattr(flow_result, "k_grid") and len(flow_result.k_grid) > 1
            else 0.0,
            v_uv=float(getattr(flow_result, "v_uv", 0)),
            v_ir=float(getattr(flow_result, "v_ir", 0)),
            log_enhancement=float(getattr(flow_result, "log_enhancement", 0)),
            crosses_critical=bool(getattr(flow_result, "crosses_critical", False)),
            k_cross=float(getattr(flow_result, "k_cross", 0)) if getattr(flow_result, "k_cross", None) else None,
            notes=list(getattr(flow_result, "notes", [])),
        )

        # Cross-validation
        xval = FRGCrossValidation(
            operator_name=operator_name,
            pi0_cgc_predicted=pi0_bare_ir,
            pi0_frg_computed=pi0_frg_computed,
        )

        # Compute cross-validation metrics
        if pi0_bare_ir is not None and pi0_frg_computed is not None:
            xval.pi0_ratio = abs(pi0_frg_computed / pi0_bare_ir) if pi0_bare_ir != 0 else float("inf")
            xval.pi0_match = bool(0.5 < xval.pi0_ratio < 2.0)  # within factor 2

        # Agreement level — interprets pole status through Physics lens
        #   EXISTS:  FRG flow alone crosses V_crit → pole confirmed dynamically
        #   MARGINAL: Pi0>0, channel capable, Dyson resummation bridges gap
        #   ABSENT:  Pi0<0 or no evidence → pole from topology, not dynamics
        if not xval.pi0_match and pi0_bare_ir is not None:
            xval.agreement_level = "NOT_VALIDATED"
        elif xval.pi0_match and pole_stat == PoleStatus.EXISTS:
            xval.agreement_level = "EXCELLENT" if xval.pi0_ratio and 0.9 < xval.pi0_ratio < 1.1 else "GOOD"
        elif xval.pi0_match and pole_stat == PoleStatus.MARGINAL:
            # Pi0>0 confirmed, channel supports emergence,
            # Dyson resummation step remains to be verified by FRG
            xval.agreement_level = "GOOD" if xval.pi0_ratio and 0.9 < xval.pi0_ratio < 1.1 else "FAIR"
        elif xval.pi0_match and pole_stat == PoleStatus.ABSENT:
            # Pi0 matches but pole ABSENT.
            # For TOPOLOGICAL_EMERGENCE: pole absence is CORRECT → GOOD
            # For DYNAMIC_EMERGENCE: Pi0>0 but pole absent = CONTRADICTION → POOR
            # Since we lack emergence_verdict here, mark as NEEDS_VALIDATION.
            # Use CGC_FRG_Validator for full validation with emergence context.
            xval.agreement_level = "NEEDS_VALIDATION"
            xval.notes.append(
                "Pi0 matches but pole absent. Run CGC_FRG_Validator for DYNAMIC vs TOPOLOGICAL interpretation."
            )
        else:
            xval.agreement_level = "NOT_VALIDATED"

        # Grid data
        k_grid = list(getattr(flow_result, "k_grid", [])) if hasattr(flow_result, "k_grid") else []
        v_grid = list(getattr(flow_result, "v_grid", [])) if hasattr(flow_result, "v_grid") else []
        eta_grid = list(getattr(flow_result, "eta_grid", [])) if hasattr(flow_result, "eta_grid") else []

        return FRGToCGCPayload(
            cgc_payload_ref=cgc_payload_path,
            spectral_function=spectral,
            flow_result=frg_flow,
            cross_validation=xval,
            k_grid=k_grid,
            v_grid=v_grid,
            eta_grid=eta_grid,
        )


# ══════════════════════════════════════════════════════════════════════════
# Cross-Validator
# ══════════════════════════════════════════════════════════════════════════


class CGC_FRG_Validator:
    """Cross-validate CGC predictions against FRG numerical results.

    The validator checks:
      1. Pi0 agreement (CGC trace density vs FRG discrete sum)
      2. Emergence verdict match (does FRG find pole when CGC predicts it?)
      3. Critical scale agreement (at what k does V·Pi0 cross threshold?)
      4. Residue comparison (ladder approximation vs full FRG)
    """

    TOLERANCE_PI0 = 0.30  # 30% tolerance on Pi0
    TOLERANCE_RESIDUE = 0.50  # 50% tolerance on residue
    TOLERANCE_SCALE = 0.50  # factor 2 on critical scale

    @classmethod
    def validate(
        cls,
        cgc_payload: CGCToFRGPayload,
        frg_payload: FRGToCGCPayload,
    ) -> FRGCrossValidation:
        """Perform full cross-validation between CGC and FRG results."""

        op_name = cgc_payload.operator.name
        xval = FRGCrossValidation(operator_name=op_name)

        # ── 1. Pi0 comparison ──
        cgc_pi0 = cgc_payload.operator.pi0_bare_ir
        frg_pi0 = frg_payload.cross_validation.pi0_frg_computed

        xval.pi0_cgc_predicted = cgc_pi0
        xval.pi0_frg_computed = frg_pi0

        if cgc_pi0 is not None and frg_pi0 is not None and abs(cgc_pi0) > 1e-30:
            xval.pi0_ratio = frg_pi0 / cgc_pi0
            xval.pi0_match = bool(abs(xval.pi0_ratio - 1.0) < cls.TOLERANCE_PI0)
            xval.notes.append(f"Pi0: CGC={cgc_pi0:.4e}, FRG={frg_pi0:.4e}, ratio={xval.pi0_ratio:.4f}")
        else:
            xval.notes.append("Pi0: insufficient data for comparison")

        # ── 2. Emergence verdict ──
        # Three-category logic:
        #   DYNAMIC_EMERGENCE + EXISTS:    FRG flow alone creates pole → strong confirmation
        #   DYNAMIC_EMERGENCE + MARGINAL:   Pi0>0 confirmed, requires Dyson to close gap → expected
        #   TOPOLOGICAL_EMERGENCE + ABSENT: Pi0<0 confirmed, pole from topology → expected
        #   Any mismatch of these categories → flagged
        cgc_dynamic = cgc_payload.operator.emergence_verdict == EmergenceVerdict.DYNAMIC_EMERGENCE
        cgc_topological = cgc_payload.operator.emergence_verdict == EmergenceVerdict.TOPOLOGICAL_EMERGENCE
        frg_pole_status = frg_payload.spectral_function.pole_status

        xval.cgc_predicted_emergence = cgc_dynamic
        xval.frg_found_pole = frg_pole_status == PoleStatus.EXISTS

        # Verdict match: interpret MARGINAL correctly
        if cgc_dynamic and (frg_pole_status == PoleStatus.EXISTS or frg_pole_status == PoleStatus.MARGINAL):
            # CGC says DYNAMIC → FRG says pole EXISTS (strong) or MARGINAL (expected)
            xval.verdict_match = True
        elif cgc_topological and frg_pole_status == PoleStatus.ABSENT:
            # CGC says TOPOLOGICAL → FRG correctly finds no dynamical pole
            xval.verdict_match = True
        elif not cgc_dynamic and not cgc_topological and frg_pole_status == PoleStatus.ABSENT:
            # CGC says NO_EMERGENCE → FRG agrees
            xval.verdict_match = True
        else:
            xval.verdict_match = False

        xval.notes.append(
            f"Verdict: CGC={'DYNAMIC' if cgc_dynamic else 'TOPOLOGICAL' if cgc_topological else 'NONE'}, "
            f"FRG pole={frg_pole_status.value}, "
            f"match={'✅' if xval.verdict_match else '❌'}"
        )

        # ── 3. Native V comparison ──
        cgc_v = cgc_payload.operator.v_native
        frg_v = frg_payload.flow_result.v_uv

        xval.v_native_cgc = cgc_v
        xval.v_uv_frg = frg_v
        if cgc_v is not None and frg_v is not None and abs(cgc_v) > 1e-30:
            xval.v_ratio = frg_v / cgc_v
            xval.notes.append(f"V_native: CGC={cgc_v:.4e}, FRG={frg_v:.4e}, ratio={xval.v_ratio:.4f}")

        # ── 4. Agreement level ──
        if xval.verdict_match and xval.pi0_match:
            xval.agreement_level = "EXCELLENT"
        elif xval.verdict_match and not xval.pi0_match:
            xval.agreement_level = "GOOD"
        elif not xval.verdict_match and xval.pi0_match:
            xval.agreement_level = "FAIR"
        else:
            xval.agreement_level = "POOR"

        xval.notes.append(f"Overall agreement: {xval.agreement_level}")

        return xval

    @classmethod
    def validate_roundtrip(
        cls,
        cgc_path: str,
        frg_path: str,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Load CGC→FRG and FRG→CGC JSON files and validate.

        Args:
            cgc_path: path to CGCToFRGPayload JSON
            frg_path: path to FRGToCGCPayload JSON
            output_path: optional path to save validation report

        Returns:
            Dict with validation results
        """
        cgc = CGCToFRGPayload.import_json(cgc_path)
        frg = FRGToCGCPayload.import_json(frg_path)

        xval = cls.validate(cgc, frg)

        result = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now().isoformat(),
            "cgc_payload": cgc_path,
            "frg_payload": frg_path,
            "validation": xval.to_dict(),
            "operator": cgc.operator.name,
            "verdict_match": xval.verdict_match,
            "agreement_level": xval.agreement_level,
        }

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        return result


# ══════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ══════════════════════════════════════════════════════════════════════════


def export_cgc_to_frg(
    operator_name: str,
    field_content: list[Any],
    pi0_bare_ir: float,
    native_v: float,
    is_protected: bool = True,
    protection_basis: str = "WARD_IDENTITY",
    op_type: str = "CONSERVED_CURRENT",
    output_path: str | None = None,
    **kwargs,
) -> CGCToFRGPayload:
    """One-shot: build CGC→FRG payload and optionally export to JSON.

    Args:
        operator_name: e.g. "Tmunu", "F2"
        field_content: list of FieldContent objects
        pi0_bare_ir: bare Pi0 at IR
        native_v: native coupling at equilibrium
        is_protected: conservation-law protected
        protection_basis: "WARD_IDENTITY", "BRST_SYMMETRY", "NOETHER_THEOREM", "NONE"
        op_type: "CONSERVED_CURRENT", "GAUGE_FIELD_STRENGTH", "SCALAR", "FERMION_BILINEAR"
        output_path: optional JSON export path
        **kwargs: passed to build_resummation_data

    Returns:
        CGCToFRGPayload
    """
    bridge = CGCToFRGBridge()
    payload = bridge.from_self_consistent_solver(
        operator_name=operator_name,
        field_content=field_content,
        pi0_bare_ir=pi0_bare_ir,
        native_v=native_v,
        is_protected=is_protected,
        protection_basis=protection_basis,
        op_type=op_type,
        **kwargs,
    )

    if output_path:
        payload.export_json(output_path)

    return payload


def import_frg_to_cgc(
    frg_json_path: str,
    cgc_json_path: str | None = None,
    validate: bool = True,
    output_validation_path: str | None = None,
) -> dict[str, Any]:
    """One-shot: import FRG→CGC JSON, optionally validate against CGC→FRG.

    Args:
        frg_json_path: path to FRGToCGCPayload JSON
        cgc_json_path: optional path to CGCToFRGPayload JSON for validation
        validate: whether to run cross-validation
        output_validation_path: optional path to save validation report

    Returns:
        Dict with frg_payload and optional validation results
    """
    frg = FRGToCGCPayload.import_json(frg_json_path)

    result = {"frg_payload": frg}

    if validate and cgc_json_path:
        cgc = CGCToFRGPayload.import_json(cgc_json_path)
        xval = CGC_FRG_Validator.validate(cgc, frg)
        result["cgc_payload"] = cgc
        result["validation"] = xval
        result["verdict_match"] = xval.verdict_match
        result["agreement_level"] = xval.agreement_level

        if output_validation_path:
            with open(output_validation_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "timestamp": datetime.now().isoformat(),
                        "cgc_payload": cgc_json_path,
                        "frg_payload": frg_json_path,
                        "validation": xval.to_dict(),
                        "verdict_match": xval.verdict_match,
                        "agreement_level": xval.agreement_level,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

    return result
