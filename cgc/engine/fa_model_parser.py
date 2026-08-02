"""
FeynArts Model File Parser
==========================

Parses FeynArts model files (SM.mod, Lorentz.gen) to extract
the Standard Model particle content and vertex couplings.

FeynArts model files are the community-standard specification
of the SM field-theory content. Using them as the data source
replaces hardcoded SM field definitions with an external,
independently-validated reference.

IRON LAWS:
  ZFP: SM content from FeynArts SM.mod (community standard, not ours).
  RH:  FeynArts model files are the single source of truth.
  RS:  All extracted data is verifiable against the source files.
  NDI: No human-assigned particle labels — all from .mod/.gen files.

Reference:
  T. Hahn, "Generating Feynman diagrams and amplitudes with
  FeynArts 3", Comput. Phys. Commun. 140 (2001) 418–431.
  Model files from: https://github.com/FeynCalc/feynarts
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# Data Model
# ═══════════════════════════════════════════════════════════════


class FAParticleType(Enum):
    """FeynArts generic particle types (defined in Lorentz.gen)."""

    F = "F"  # Fermion (Dirac/Majorana)
    V = "V"  # Vector boson
    S = "S"  # Scalar
    U = "U"  # Ghost
    SV = "SV"  # Mixed scalar-vector


class FAFieldCategory(Enum):
    """CGC-relevant classification of FeynArts fields."""

    WEYL_FERMION = "weyl_fermion"
    GAUGE_BOSON = "gauge_boson"
    REAL_SCALAR = "real_scalar"
    COMPLEX_SCALAR = "complex_scalar"
    GHOST = "ghost"


@dataclass
class FAParticleClass:
    """One particle class from M$ClassesDescription."""

    generic_type: FAParticleType
    class_index: int
    mass_expr: str = "0"
    self_conjugate: bool = False
    quantum_numbers: dict[str, str] = field(default_factory=dict)
    propagator_label: str = ""
    generation_range: int = 1
    is_mixed: bool = False

    @property
    def cgc_category(self) -> FAFieldCategory:
        if self.generic_type == FAParticleType.F:
            return FAFieldCategory.WEYL_FERMION
        if self.generic_type == FAParticleType.V:
            return FAFieldCategory.GAUGE_BOSON
        if self.generic_type == FAParticleType.S:
            if self.quantum_numbers.get("Charge") not in (None, "0", ""):
                return FAFieldCategory.COMPLEX_SCALAR
            return FAFieldCategory.REAL_SCALAR
        if self.generic_type == FAParticleType.U:
            return FAFieldCategory.GHOST
        return FAFieldCategory.REAL_SCALAR

    @property
    def cgc_propagator_label(self) -> str:
        mapping = {
            FAFieldCategory.WEYL_FERMION: "ψ",
            FAFieldCategory.GAUGE_BOSON: "A",
            FAFieldCategory.REAL_SCALAR: "φ",
            FAFieldCategory.COMPLEX_SCALAR: "φ±",
            FAFieldCategory.GHOST: "c",
        }
        return mapping.get(self.cgc_category, "?")

    @property
    def cgc_propagator_sign(self) -> int:
        if self.cgc_category == FAFieldCategory.GHOST:
            return -1
        return 1

    @property
    def dof_per_species(self) -> int:
        if self.generic_type == FAParticleType.F:
            return 4  # Weyl fermion (2-component × 2 helicity)
        if self.generic_type == FAParticleType.V:
            return 2  # transverse
        if self.generic_type == FAParticleType.S or self.generic_type == FAParticleType.U:
            return 1
        return 1


@dataclass
class FAVertex:
    """One interaction vertex from M$CouplingMatrices."""

    fields: list[str]  # e.g. ["F[2]", "-F[2]", "V[1]"]
    coupling_name: str
    lorentz_structure: str = ""

    @property
    def field_types(self) -> list[str]:
        """Extract generic field types: F, V, S, U."""
        return [re.sub(r"[^FVS]", "", f.split("[")[0].lstrip("-")) for f in self.fields]

    @property
    def valence(self) -> int:
        return len(self.fields)


@dataclass
class FeynArtsModel:
    """Complete parsed FeynArts model."""

    source_file: str = ""
    particles: list[FAParticleClass] = field(default_factory=list)
    vertices: list[FAVertex] = field(default_factory=list)

    def get_cgc_coupled_fields(self) -> dict[str, list[str]]:
        """Map field category → list of particle class names."""
        result: dict[str, list[str]] = {}
        for p in self.particles:
            cat = p.cgc_category.value
            if cat not in result:
                result[cat] = []
            result[cat].append(f"{p.generic_type.value}[{p.class_index}]")
        return result

    def count_by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.particles:
            cat = p.cgc_category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts


# ═══════════════════════════════════════════════════════════════
# Parser
# ═══════════════════════════════════════════════════════════════


def parse_feynarts_model(sm_mod_path: str | Path) -> FeynArtsModel:
    """Parse a FeynArts .mod file into structured model data.

    Handles the Classes model format (SM.mod, MSSM.mod, etc.).
    Does NOT require Mathematica — pure Python parsing of the
    text-based model file format.

    Extracts:
      - M$ClassesDescription → particle classes
      - M$CouplingMatrices   → interaction vertices
      - M$LastModelRules     → additional rules

    Args:
        sm_mod_path: Path to the .mod file.

    Returns:
        FeynArtsModel with particles and vertices populated.
    """
    path = Path(sm_mod_path)
    content = path.read_text(encoding="utf-8", errors="replace")

    # Strip Mathematica comments
    content = _strip_comments(content)

    model = FeynArtsModel(source_file=str(path))

    # ── Parse M$ClassesDescription ──
    classes_section = _extract_section(content, r"M\$ClassesDescription\s*=\s*\{")
    if classes_section:
        model.particles = _parse_classes(classes_section)

    # ── Parse M$CouplingMatrices ──
    coupling_section = _extract_section(content, r"M\$CouplingMatrices\s*=\s*\{")
    if coupling_section:
        model.vertices = _parse_vertices(coupling_section)

    return model


def _strip_comments(content: str) -> str:
    """Remove Mathematica (* ... *) comments (handling nesting)."""
    result = []
    depth = 0
    i = 0
    while i < len(content):
        if content[i : i + 2] == "(*":
            depth += 1
            i += 2
        elif depth > 0 and content[i : i + 2] == "*)":
            depth -= 1
            i += 2
        elif depth == 0:
            result.append(content[i])
            i += 1
        else:
            i += 1
    return "".join(result)


def _extract_section(content: str, start_pattern: str) -> str:
    """Extract a braced section starting with start_pattern.
    Returns the content between the outermost braces.
    """
    m = re.search(start_pattern, content)
    if not m:
        return ""
    start = m.end() - 1  # include the opening {
    depth = 0
    for i in range(start, len(content)):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                return content[start + 1 : i]
    return ""


def _parse_classes(section: str) -> list[FAParticleClass]:
    """Parse M$ClassesDescription into FAParticleClass list."""
    particles = []
    # Each class starts with a type spec like F[1], V[2], S[1], U[1]
    # Find class boundaries by looking for these patterns at top level
    re.compile(
        r"""
        ^\s*
        (F|V|S|U|SV|Mix\[[FVSU]+,[FVSU]+\])\s*\[(\d+)\]
        """,
        re.MULTILINE | re.VERBOSE,
    )

    # Actually, FeynArts classes have a specific structure.
    # Let's parse more carefully — split by class entries.
    # Each entry: F[n] == { ... } or V[n] == { ... }

    entries = _split_class_entries(section)

    for entry in entries:
        pc = _parse_one_class(entry)
        if pc is not None:
            particles.append(pc)

    return particles


def _split_class_entries(section: str) -> list[str]:
    """Split the classes section into individual class entries."""
    entries = []
    # Match class headers: F[X] == { or V[X] == { or S[X] == {
    current: list[str] = []
    brace_depth = 0

    for line in section.split("\n"):
        stripped = line.strip()

        # Check for class header
        header_match = re.match(
            r"(Mix\[[FVSU]+,[FVSU]+\]\s*\[\d+\]|Mix\[[FVSU]+\]\s*\[\d+\]|[FVSU]+\s*\[\d+\])\s*==\s*\{", stripped
        )

        if header_match and brace_depth == 0:
            # Save previous entry
            if current:
                entries.append("\n".join(current))
            current = [line]
            brace_depth = 1
            # Count additional braces on this line
            brace_depth += stripped.count("{") - 1  # -1 for the opening {
            brace_depth -= stripped.count("}")
            if brace_depth <= 0:
                entries.append("\n".join(current))
                current = []
        elif current:
            current.append(line)
            brace_depth += stripped.count("{")
            brace_depth -= stripped.count("}")
            if brace_depth <= 0:
                entries.append("\n".join(current))
                current = []

    if current:
        entries.append("\n".join(current))

    return entries


def _parse_one_class(entry: str) -> FAParticleClass | None:
    """Parse a single class entry block."""
    # Detect generic type and index
    type_match = re.match(
        r"^\s*(Mix\[[FVSU]+,[FVSU]+\]\s*\[(\d+)\]|Mix\[[FVSU]+\]\s*\[(\d+)\]|([FVSU]+)\s*\[(\d+)\])", entry
    )
    if not type_match:
        return None

    raw_type = type_match.group(0).strip()
    is_mixed = "Mix" in raw_type

    # Extract generic type letter
    type_letter = ""
    for t in ["F", "V", "S", "U"]:
        if t in raw_type.split("[")[0]:
            type_letter = t
            break

    # Extract index
    idx_match = re.search(r"\[(\d+)\]", raw_type)
    idx = int(idx_match.group(1)) if idx_match else 0

    try:
        generic_type = FAParticleType(type_letter) if type_letter else FAParticleType.S
    except ValueError:
        generic_type = FAParticleType.S

    # Extract properties
    mass_expr = "0"
    mass_match = re.search(r"Mass\s*->\s*([^,\}]+)", entry)
    if mass_match:
        mass_expr = mass_match.group(1).strip()

    self_conj = False
    sc_match = re.search(r"SelfConjugate\s*->\s*(True|False)", entry)
    if sc_match:
        self_conj = sc_match.group(1) == "True"

    quantum_numbers: dict[str, str] = {}
    qn_match = re.search(r"QuantumNumbers\s*->\s*\{([^}]+)\}", entry)
    if qn_match:
        qn_text = qn_match.group(1)
        for qn in qn_text.split(","):
            qn = qn.strip()
            parts = qn.split(None, 1)
            if len(parts) >= 2:
                quantum_numbers[parts[1]] = parts[0]
            elif parts:
                quantum_numbers[parts[0]] = "1"

    prop_label = ""
    pl_match = re.search(r'PropagatorLabel\s*->\s*(ComposedChar\[[^\]]+\]|"[^"]*")', entry)
    if pl_match:
        prop_label = pl_match.group(1)

    # Generation range
    gen_range = 1
    gen_match = re.search(r"Indices\s*->\s*\{Index\[Generation[,\s]*\{(\d+),\s*(\d+)\}\]", entry)
    if gen_match:
        gen_range = int(gen_match.group(2)) - int(gen_match.group(1)) + 1

    return FAParticleClass(
        generic_type=generic_type,
        class_index=idx,
        mass_expr=mass_expr,
        self_conjugate=self_conj,
        quantum_numbers=quantum_numbers,
        propagator_label=prop_label,
        generation_range=gen_range,
        is_mixed=is_mixed,
    )


def _parse_vertices(section: str) -> list[FAVertex]:
    """Parse M$CouplingMatrices into FAVertex list."""
    vertices = []

    # Split by C[...] patterns
    segments = re.split(r"(?=C\s*\[)", section)

    for seg in segments:
        seg = seg.strip()
        if not seg.startswith("C"):
            continue

        # Bracket-matching: find the matching ] for C[...]
        # C[ -V[3], V[3] ]  -> fields are between the outermost [ and ]
        bracket_start = seg.find("[")
        if bracket_start < 0:
            continue
        depth = 0
        bracket_end = -1
        for i in range(bracket_start, len(seg)):
            if seg[i] == "[":
                depth += 1
            elif seg[i] == "]":
                depth -= 1
                if depth == 0:
                    bracket_end = i
                    break

        if bracket_end < 0:
            continue

        fields_str = seg[bracket_start + 1 : bracket_end]
        fields = _parse_vertex_fields(fields_str)

        # Extract coupling name from after ==
        coupling_name = ""
        eq_idx = seg.find("==")
        if eq_idx >= 0:
            after_eq = seg[eq_idx + 2 :].strip()
            star_idx = after_eq.find("*")
            coupling_name = after_eq[:star_idx].strip() if star_idx >= 0 else after_eq[:80].strip()

        if fields:
            vertices.append(
                FAVertex(
                    fields=fields,
                    coupling_name=coupling_name or "unknown",
                )
            )

    return vertices


def _parse_vertex_fields(fields_str: str) -> list[str]:
    """Parse the field list from C[...] syntax.

    Handles:
      "S[1], S[1], S[1], S[1]"
      "-F[2, {j1}], F[2, {j2}], V[1]"
      "S[3], -V[3], V[2]"
    """
    fields = []
    depth = 0
    current = []

    for ch in fields_str:
        if ch in "{[(":
            depth += 1
            current.append(ch)
        elif ch in "}])":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            fields.append("".join(current).strip())
            current = []
        else:
            current.append(ch)

    if current:
        fields.append("".join(current).strip())

    return fields


# ═══════════════════════════════════════════════════════════════
# Integration Helpers
# ═══════════════════════════════════════════════════════════════


def load_sm_model(mod_path: str | Path | None = None) -> FeynArtsModel:
    """Load the Standard Model from FeynArts model files.

    If mod_path is not specified, looks for SM.mod in:
      1. cgc/data/feynarts/SM.mod
      2. The FEYNARTS_PATH environment variable
      3. The current working directory
    """
    if mod_path is not None:
        return parse_feynarts_model(mod_path)

    import os

    search_paths = [
        Path(__file__).parent.parent / "data" / "feynarts" / "SM.mod",
        Path(os.environ.get("FEYNARTS_PATH", "")) / "SM.mod",
        Path("SM.mod"),
    ]

    for p in search_paths:
        if p.exists():
            return parse_feynarts_model(p)

    raise FileNotFoundError(
        "SM.mod not found. Download from:\n"
        "  https://raw.githubusercontent.com/FeynCalc/feynarts/master/Models/SM.mod\n"
        "Save to cgc/data/feynarts/SM.mod"
    )


def model_to_cgc_summary(model: FeynArtsModel) -> str:
    """Produce a human-readable summary of parsed model content."""
    lines = [f"FeynArts Model: {model.source_file}"]
    lines.append(f"  Particles: {len(model.particles)} classes")
    lines.append(f"  Vertices:  {len(model.vertices)} interactions")
    lines.append("")

    # Particle table
    lines.append("  Particle Classes:")
    lines.append(f"  {'Type':<8} {'Idx':<4} {'Category':<18} {'Mass':<10} {'#Gen':<5} {'SC':<4} {'Label'}")
    lines.append("  " + "-" * 80)
    for p in model.particles:
        mass_short = p.mass_expr[:10] if len(p.mass_expr) <= 10 else p.mass_expr[:9] + "…"
        lines.append(
            f"  {p.generic_type.value:<8} [{p.class_index:<2}] {p.cgc_category.value:<18} "
            f"{mass_short:<10} {p.generation_range:<5} {str(p.self_conjugate):<4} "
            f"{p.propagator_label[:30]}"
        )

    # Category summary
    lines.append("")
    lines.append("  Category Summary:")
    for cat, count in model.count_by_category().items():
        lines.append(f"    {cat}: {count}")

    # Vertex summary
    lines.append("")
    lines.append(f"  Vertices ({len(model.vertices)} total):")
    for v in model.vertices[:20]:
        fields_str = ", ".join(v.fields)
        lines.append(f"    C[{fields_str}] = {v.coupling_name[:50]}")
    if len(model.vertices) > 20:
        lines.append(f"    ... and {len(model.vertices) - 20} more")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    mod_path = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        model = load_sm_model(mod_path)
        print(model_to_cgc_summary(model))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
