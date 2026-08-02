"""CGC↔FRG Automated Interface — Gap 4.

Bidirectional JSON-based data exchange between CGC (Conservation-Guided
Classification) and FRG (Functional Renormalization Group) engines.

CGC → FRG: operator classification, field content, spectrum parameters,
            resummation structure → tells FRG what to compute
FRG → CGC: spectral function, V(k) flow, criticality status,
            cross-validation → CGC validates its predictions

Schema version: 1.0.0
Author: CGC-FRG Integration
Date: 2026-07-30
"""

from .bridge import (
    CGC_FRG_Validator,
    CGCToFRGBridge,
    FRGToCGCBridge,
    export_cgc_to_frg,
    import_frg_to_cgc,
)
from .schema import (
    # schema metadata
    SCHEMA_VERSION,
    CGCFieldContent,
    # CGC → FRG
    CGCOperatorSpec,
    CGCResummationData,
    CGCSpectrumParams,
    CGCToFRGPayload,
    FRGCrossValidation,
    FRGFlowResult,
    # FRG → CGC
    FRGSpectralFunction,
    FRGToCGCPayload,
    get_json_schema,
)
from .semantic_validate import (
    run_semantic_validation,
    validate_pi0_agreement,
    validate_v_flow,
)

__all__ = [
    "CGCOperatorSpec",
    "CGCFieldContent",
    "CGCSpectrumParams",
    "CGCResummationData",
    "CGCToFRGPayload",
    "FRGSpectralFunction",
    "FRGFlowResult",
    "FRGCrossValidation",
    "FRGToCGCPayload",
    "SCHEMA_VERSION",
    "get_json_schema",
    "CGCToFRGBridge",
    "FRGToCGCBridge",
    "CGC_FRG_Validator",
    "export_cgc_to_frg",
    "import_frg_to_cgc",
    "validate_pi0_agreement",
    "validate_v_flow",
    "run_semantic_validation",
]
