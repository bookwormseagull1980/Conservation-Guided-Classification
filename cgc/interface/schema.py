"""CGC↔FRG JSON Schema — v1.0.0.

Defines the canonical data structures for bidirectional exchange between
the CGC classification engine and the FRG numerical engine.

Design principles:
  1. All physical quantities carry explicit units (values in GeV^n)
  2. Metadata fields enable automated provenance tracking
  3. Optional fields marked as Optional[type] = None
  4. Every dataclass has to_dict() / from_dict() for JSON round-trip
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum

import cgc

SCHEMA_VERSION = cgc.__version__


# ══════════════════════════════════════════════════════════════════════════
# Shared Enums
# ══════════════════════════════════════════════════════════════════════════


class FieldSpecies(str, Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    SPINOR = "spinor"
    TENSOR_TT = "tensor_TT"


class OperatorType(str, Enum):
    CONSERVED_CURRENT = "CONSERVED_CURRENT"  # Tμν, Jμ
    GAUGE_FIELD_STRENGTH = "GAUGE_FIELD_STRENGTH"  # F^2
    SCALAR = "SCALAR"  # φ^4, φ^2
    FERMION_BILINEAR = "FERMION_BILINEAR"  # ψ̄ψ


class ProtectionBasis(str, Enum):
    WARD_IDENTITY = "WARD_IDENTITY"
    BRST_SYMMETRY = "BRST_SYMMETRY"
    NOETHER_THEOREM = "NOETHER_THEOREM"
    NONE = "NONE"


class EmergenceVerdict(str, Enum):
    """CGC classification verdict for an operator."""

    DYNAMIC_EMERGENCE = "DYNAMIC_EMERGENCE"
    """Conservation-protected, Pi0>0, ladder accumulates → pole CAN form"""
    TOPOLOGICAL_EMERGENCE = "TOPOLOGICAL_EMERGENCE"
    """Conservation-protected but Pi0<0 → pole from topology, not dynamics"""
    NO_EMERGENCE = "NO_EMERGENCE"
    """Not protected → injection suppressed → no pole"""
    UNCERTAIN = "UNCERTAIN"
    """Edge case: protected but Pi0~0 or ambiguous"""


class PoleStatus(str, Enum):
    EXISTS = "EXISTS"
    ABSENT = "ABSENT"
    MARGINAL = "MARGINAL"
    NOT_COMPUTED = "NOT_COMPUTED"


# ══════════════════════════════════════════════════════════════════════════
# CGC → FRG Payload
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class CGCOperatorSpec:
    """Operator specification from CGC classification.

    This is the primary output of the CGC pipeline for one operator.
    It tells the FRG engine: what operator, what field content,
    what spectrum, and what the resummation structure predicts.
    """

    name: str
    """Human-readable name, e.g. 'Tμν spin-2 (TT projection)'"""

    operator_type: OperatorType
    """CGC operator type classification"""

    # Protection
    is_protected: bool
    protection_basis: ProtectionBasis
    matrix_element_nonzero: bool
    emergence_verdict: EmergenceVerdict

    # Pi0 sign (critical for Dyson direction)
    pi0_sign: int = 0
    """+1 = boson-dominated (amplification), -1 = fermion-dominated (suppression),
    0 = zero or unknown"""

    pi0_bare_ir: float | None = None
    """Bare Pi0 at IR (dimensionless), positive→amplification, negative→suppression"""

    # Critical condition
    lambda_crit: float | None = None
    """Critical coupling from the cubic-vertex condition: 4/27 ≈ 0.148
    (the standard O(N) cubic-vertex critical value)."""

    v_native: float | None = None
    """Native (bare) coupling value at equilibrium geometry"""

    v_crit_needed: float | None = None
    """V needed to reach criticality: v_crit = lambda_crit / Pi0_bare_IR"""

    # Gap
    gap_to_criticality: float | None = None
    """v_crit / v_native — how many times native V must be enhanced"""

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(asdict(self).items())
        d["operator_type"] = self.operator_type.value
        d["protection_basis"] = self.protection_basis.value
        d["emergence_verdict"] = self.emergence_verdict.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CGCOperatorSpec:
        d = dict(d)
        d["operator_type"] = OperatorType(d["operator_type"])
        d["protection_basis"] = ProtectionBasis(d["protection_basis"])
        d["emergence_verdict"] = EmergenceVerdict(d["emergence_verdict"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CGCFieldContent:
    """One SM field species contributing on RP³.

    This mirrors frg_flow_rp3.FieldContent but with JSON-safe types.
    """

    name: str
    field_type: FieldSpecies
    n_species: int
    dof_per_species: int
    mass_gev: float = 0.0
    coupling_sq: float = 1.0
    """Dimensionless coupling^2 to the operator at equilibrium geometry"""

    @property
    def total_dof(self) -> int:
        return self.n_species * self.dof_per_species

    def to_dict(self) -> dict:
        d = asdict(self)
        d["field_type"] = self.field_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CGCFieldContent:
        d = dict(d)
        d["field_type"] = FieldSpecies(d["field_type"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CGCSpectrumParams:
    """RP³ discrete spectrum parameters.

    Eigenvalue formulas from Camporesi (1990), verified 38/38 tests.
    """

    L_rp3: float = 2.44
    """RP³ radius at sigma_G (dimensionless, in M_P^-1 units)"""

    M_P: float = 2.4353e18
    """Reduced Planck mass [GeV]"""

    M_CURV: float = 0.0
    """Curvature mass scale M_P/L [GeV]. Computed if zero."""

    T_flavor: int = 5
    """SU(N_F) flavor number (determines EC torsion structure)"""

    # Eigenvalue formulas (human-readable, for reference)
    scalar_lambda: str = "J(J+2) * M_CURV^2,  J=0,2,4,..."
    vector_lambda: str = "(J+1)^2 * M_CURV^2,  J=1,3,5,..."
    spinor_lambda: str = "(QN+3/2)^2 * M_CURV^2,  QN=0,2,4,..."
    tensor_TT_lambda: str = "J(J+2) * M_CURV^2,  J=2,4,..."

    # Degeneracies
    scalar_degeneracy: str = "(J+1)^2"
    vector_degeneracy: str = "2J(J+2)"
    spinor_degeneracy: str = "(QN+1)(QN+2)"
    tensor_TT_degeneracy: str = "(J+1)^2"

    # Mode counts below M_P (validated 2026-07-29)
    n_scalar_modes_below_MP: int = 1
    n_vector_modes_below_MP: int = 1
    n_spinor_modes_below_MP: int = 1
    n_tensor_TT_modes_below_MP: int = 1

    def __post_init__(self):
        if self.M_CURV == 0.0:
            self.M_CURV = self.M_P / self.L_rp3

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CGCSpectrumParams:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CGCResummationData:
    """Resummation structure from CGC ladder analysis.

    This encodes the diagram topology → injection → ladder resummation chain.
    """

    operator_name: str

    # Diagram counts
    n_total_diagrams: int = 0
    n_q0_diagrams: int = 0
    n_ladder_diagrams: int = 0
    n_single_bubble: int = 0

    # Resummation formula
    resummation_formula: str = ""
    """Human-readable: Pi_resum = Pi0 / (1 - V*Pi0)"""

    # Self-consistent Dyson enhancement
    dyson_y_crit: float = 0.14814814814814814
    """Critical V*Pi0 from cubic vertex: 4/27"""

    bcs_integral_at_native_V: float | None = None
    """BCS-type self-consistency integral value at native V"""

    has_bcs_solution: bool = False
    """Whether BCS equation has a self-consistent solution"""

    # Enhancement budget
    spectral_enhancement: float | None = None
    """Max spectral enhancement factor from mode compression"""

    dyson_amplification: float | None = None
    """Dyson self-consistent amplification factor"""

    total_enhancement: float | None = None
    """Product of all enhancement factors"""

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CGCResummationData:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CGCToFRGPayload:
    """Complete CGC→FRG payload: everything FRG needs to compute.

    This is the canonical output of the CGC pipeline packaged for
    consumption by the FRG numerical engine.
    """

    # Metadata
    schema_version: str = SCHEMA_VERSION
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "CGC Pipeline"
    pipeline_version: str = "0.1.0"

    # Core classification
    operator: CGCOperatorSpec = field(
        default_factory=lambda: CGCOperatorSpec(
            name="unknown",
            operator_type=OperatorType.SCALAR,
            is_protected=False,
            protection_basis=ProtectionBasis.NONE,
            matrix_element_nonzero=False,
            emergence_verdict=EmergenceVerdict.NO_EMERGENCE,
        )
    )

    # What to compute
    compute_instructions: list[str] = field(default_factory=list)
    """Human-readable list of FRG computations to perform"""

    # Field content for FRG trace density
    fields: list[CGCFieldContent] = field(default_factory=list)

    # Spectrum
    spectrum: CGCSpectrumParams = field(default_factory=CGCSpectrumParams)

    # Resummation structure
    resummation: CGCResummationData = field(default_factory=lambda: CGCResummationData(operator_name="unknown"))

    # FRG parameter recommendations
    frg_recommendations: dict[str, float | int | str] = field(default_factory=dict)
    """Recommended FRG parameters: k_UV, k_IR, n_grid, regulator_type, etc."""

    # Errors/warnings
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "source": self.source,
            "pipeline_version": self.pipeline_version,
            "operator": self.operator.to_dict(),
            "compute_instructions": self.compute_instructions,
            "fields": [f.to_dict() for f in self.fields],
            "spectrum": self.spectrum.to_dict(),
            "resummation": self.resummation.to_dict(),
            "frg_recommendations": self.frg_recommendations,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CGCToFRGPayload:
        return cls(
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            timestamp=d.get("timestamp", ""),
            source=d.get("source", "CGC Pipeline"),
            pipeline_version=d.get("pipeline_version", "0.1.0"),
            operator=CGCOperatorSpec.from_dict(d.get("operator", {})),
            compute_instructions=d.get("compute_instructions", []),
            fields=[CGCFieldContent.from_dict(f) for f in d.get("fields", [])],
            spectrum=CGCSpectrumParams.from_dict(d.get("spectrum", {})),
            resummation=CGCResummationData.from_dict(d.get("resummation", {})),
            frg_recommendations=d.get("frg_recommendations", {}),
            errors=d.get("errors", []),
            warnings=d.get("warnings", []),
        )

    def export_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def import_json(cls, path: str) -> CGCToFRGPayload:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ══════════════════════════════════════════════════════════════════════════
# FRG → CGC Payload
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class FRGSpectralFunction:
    """FRG-computed spectral function for one operator.

    This is the primary output of the RP3 discrete-spectrum FRG solver.
    """

    operator_name: str

    # Pole
    pole_status: PoleStatus = PoleStatus.NOT_COMPUTED
    pole_position_gev2: float | None = None
    """Pole position in GeV^2 (should be 0 for massless emergence)"""
    pole_residue: float | None = None
    """Spectral weight of the delta pole"""

    # Continuum
    continuum_threshold_gev2: float | None = None
    continuum_shape: str = ""
    """Qualitative description: 'power-law', 'exponential', 'step', etc."""

    # Total spectral density
    spectral_density_at_zero: float | None = None
    """Spectral density rho(mu^2=0), should diverge if pole exists"""

    # Notes
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pole_status"] = self.pole_status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FRGSpectralFunction:
        d = dict(d)
        d["pole_status"] = PoleStatus(d["pole_status"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FRGFlowResult:
    """FRG flow integration result for one operator."""

    operator_name: str

    # Grid
    k_uv: float = 0.0
    k_ir: float = 0.0
    n_grid: int = 500

    # Flow
    v_uv: float = 0.0
    v_ir: float = 0.0
    log_enhancement: float = 0.0

    # Beta function
    beta_sign_at_mid: str = ""
    beta_mid_value: float = 0.0

    # Criticality
    crosses_critical: bool = False
    k_cross: float | None = None
    """RG scale where V reaches lambda_crit, if any"""

    # Key scales
    v_at_M_CURV: float | None = None
    v_at_M_G: float | None = None
    v_at_1TeV: float | None = None

    # Regulator info
    regulator_type: str = "Litim"
    include_anomalous_dim: bool = True
    include_v3: bool = True

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FRGFlowResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FRGCrossValidation:
    """Cross-validation: CGC prediction vs FRG numerical result.

    This is the feedback loop: FRG results are compared against
    CGC predictions to validate (or falsify) the ladder approximation.
    """

    operator_name: str

    # Pi0 comparison
    pi0_cgc_predicted: float | None = None
    pi0_frg_computed: float | None = None
    pi0_ratio: float | None = None
    pi0_match: bool = False
    """Whether CGC Pi0 matches FRG Pi0 within tolerance"""

    # V comparison
    v_native_cgc: float | None = None
    v_uv_frg: float | None = None
    v_ratio: float | None = None

    # Criticality comparison
    cgc_predicted_emergence: bool = False
    frg_found_pole: bool = False
    verdict_match: bool = False
    """Whether CGC emergence prediction matches FRG pole finding"""

    # Pole residue comparison
    residue_cgc_predicted: float | None = None
    residue_frg_computed: float | None = None
    residue_ratio: float | None = None
    residue_match: bool = False

    # Agreement metrics
    agreement_level: str = "NOT_VALIDATED"
    """One of: EXCELLENT, GOOD, FAIR, POOR, NOT_VALIDATED"""

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FRGCrossValidation:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FRGToCGCPayload:
    """Complete FRG→CGC payload: FRG results returned for CGC validation.

    This is the canonical output of the RP3 discrete-spectrum FRG solver
    packaged for consumption by the CGC validation engine.
    """

    # Metadata
    schema_version: str = SCHEMA_VERSION
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "FRG RP3 Discrete-Spectrum Solver"
    frg_version: str = "0.1.0"

    # Reference to original CGC payload
    cgc_payload_ref: str = ""
    """Path or hash of the original CGCToFRGPayload that triggered this run"""

    # Results per operator
    spectral_function: FRGSpectralFunction = field(default_factory=lambda: FRGSpectralFunction(operator_name="unknown"))
    flow_result: FRGFlowResult = field(default_factory=lambda: FRGFlowResult(operator_name="unknown"))

    # Cross-validation
    cross_validation: FRGCrossValidation = field(default_factory=lambda: FRGCrossValidation(operator_name="unknown"))

    # Raw data (optional, for debugging)
    k_grid: list[float] = field(default_factory=list)
    v_grid: list[float] = field(default_factory=list)
    eta_grid: list[float] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "source": self.source,
            "frg_version": self.frg_version,
            "cgc_payload_ref": self.cgc_payload_ref,
            "spectral_function": self.spectral_function.to_dict(),
            "flow_result": self.flow_result.to_dict(),
            "cross_validation": self.cross_validation.to_dict(),
            "k_grid": self.k_grid,
            "v_grid": self.v_grid,
            "eta_grid": self.eta_grid,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FRGToCGCPayload:
        return cls(
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            timestamp=d.get("timestamp", ""),
            source=d.get("source", "FRG RP3 Discrete-Spectrum Solver"),
            frg_version=d.get("frg_version", "0.1.0"),
            cgc_payload_ref=d.get("cgc_payload_ref", ""),
            spectral_function=FRGSpectralFunction.from_dict(d.get("spectral_function", {})),
            flow_result=FRGFlowResult.from_dict(d.get("flow_result", {})),
            cross_validation=FRGCrossValidation.from_dict(d.get("cross_validation", {})),
            k_grid=d.get("k_grid", []),
            v_grid=d.get("v_grid", []),
            eta_grid=d.get("eta_grid", []),
            errors=d.get("errors", []),
            warnings=d.get("warnings", []),
        )

    def export_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def import_json(cls, path: str) -> FRGToCGCPayload:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ══════════════════════════════════════════════════════════════════════════
# JSON Schema (for validation)
# ══════════════════════════════════════════════════════════════════════════


def get_json_schema(direction: str = "cgc_to_frg") -> dict:
    """Generate a JSON Schema for validation of CGC↔FRG payloads.

    Args:
        direction: "cgc_to_frg" or "frg_to_cgc"

    Returns:
        JSON Schema dict (Draft-07 compatible)
    """
    if direction == "cgc_to_frg":
        return _cgc_to_frg_schema()
    return _frg_to_cgc_schema()


def _cgc_to_frg_schema() -> dict:
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": f"https://cgc-framework.org/schema/cgc_to_frg/v{SCHEMA_VERSION}",
        "title": "CGC to FRG Payload",
        "description": "Operator classification and field content for FRG computation",
        "type": "object",
        "required": ["schema_version", "operator", "fields", "spectrum"],
        "properties": {
            "schema_version": {"type": "string", "const": SCHEMA_VERSION},
            "timestamp": {"type": "string", "format": "date-time"},
            "source": {"type": "string"},
            "pipeline_version": {"type": "string"},
            "operator": {
                "type": "object",
                "required": ["name", "operator_type", "emergence_verdict"],
                "properties": {
                    "name": {"type": "string"},
                    "operator_type": {"type": "string", "enum": [e.value for e in OperatorType]},
                    "is_protected": {"type": "boolean"},
                    "protection_basis": {"type": "string", "enum": [e.value for e in ProtectionBasis]},
                    "emergence_verdict": {"type": "string", "enum": [e.value for e in EmergenceVerdict]},
                    "pi0_sign": {"type": "integer", "enum": [-1, 0, 1]},
                    "pi0_bare_ir": {"type": ["number", "null"]},
                    "lambda_crit": {"type": ["number", "null"]},
                    "v_native": {"type": ["number", "null"]},
                    "v_crit_needed": {"type": ["number", "null"]},
                    "gap_to_criticality": {"type": ["number", "null"]},
                },
            },
            "compute_instructions": {"type": "array", "items": {"type": "string"}},
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "field_type", "n_species", "dof_per_species"],
                    "properties": {
                        "name": {"type": "string"},
                        "field_type": {"type": "string", "enum": [e.value for e in FieldSpecies]},
                        "n_species": {"type": "integer", "minimum": 0},
                        "dof_per_species": {"type": "integer", "minimum": 0},
                        "mass_gev": {"type": "number", "minimum": 0},
                        "coupling_sq": {"type": "number", "minimum": 0},
                    },
                },
            },
            "spectrum": {
                "type": "object",
                "required": ["L_rp3", "M_P"],
                "properties": {
                    "L_rp3": {"type": "number", "minimum": 0},
                    "M_P": {"type": "number", "minimum": 0},
                    "M_CURV": {"type": "number", "minimum": 0},
                },
            },
            "resummation": {"type": "object"},
            "frg_recommendations": {"type": "object"},
        },
    }


def _frg_to_cgc_schema() -> dict:
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": f"https://cgc-framework.org/schema/frg_to_cgc/v{SCHEMA_VERSION}",
        "title": "FRG to CGC Payload",
        "description": "FRG numerical results returned for CGC validation",
        "type": "object",
        "required": ["schema_version", "spectral_function", "flow_result"],
        "properties": {
            "schema_version": {"type": "string", "const": SCHEMA_VERSION},
            "timestamp": {"type": "string", "format": "date-time"},
            "source": {"type": "string"},
            "frg_version": {"type": "string"},
            "cgc_payload_ref": {"type": "string"},
            "spectral_function": {
                "type": "object",
                "required": ["operator_name", "pole_status"],
                "properties": {
                    "operator_name": {"type": "string"},
                    "pole_status": {"type": "string", "enum": [e.value for e in PoleStatus]},
                    "pole_position_gev2": {"type": ["number", "null"]},
                    "pole_residue": {"type": ["number", "null"]},
                },
            },
            "flow_result": {
                "type": "object",
                "required": ["operator_name"],
                "properties": {
                    "operator_name": {"type": "string"},
                    "v_uv": {"type": "number"},
                    "v_ir": {"type": "number"},
                    "crosses_critical": {"type": "boolean"},
                },
            },
            "cross_validation": {"type": "object"},
        },
    }
