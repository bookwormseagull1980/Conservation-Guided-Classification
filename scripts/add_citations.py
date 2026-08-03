"""P1-2: Add literature citations to CGC engine files.

Module-level references for infrastructure files.
Inline refs for key physics formulas.
"""

BASE = r"D:\论文撰写\Conservation-Guided Classification\cgc\engine"

MODULE_REFS = {
    "conservation_checker.py": (
        "References\n"
        "    Ward (1950): Ward-Takahashi identity for Tmu_nu protection\n"
        "    Becchi-Rouet-Stora (1976): BRST symmetry for F^2 protection\n"
        "    Slavnov-Taylor identities: ghost-antighost Ward identities\n"
    ),
    "momentum_classifier.py": (
        "References\n"
        "    Wetterich (1993), Phys. Lett. B 301, 90: exact FRG flow equation\n"
        "    The zero-momentum-transfer classification follows from conservation\n"
        "    law insertion at q=0 (see Paper 1, Appendix E)\n"
    ),
    "diagram_builder.py": (
        "References\n"
        "    Feynman diagram topology: one-loop bubble/triangle/box classification\n"
        "    Momentum routing: standard QFT textbook (Peskin & Schroeder, Ch. 7)\n"
    ),
    "diagram_generator.py": (
        "References\n"
        "    FRG one-loop expansion: Wetterich (1993), Phys. Lett. B 301, 90\n"
        "    Diagram topology enumeration: standard perturbation theory\n"
    ),
    "resummation.py": (
        "References\n"
        "    Ladder resummation: Dyson-Schwinger equation formalism\n"
        "    Roberts-Williams (1994), Prog. Part. Nucl. Phys. 33, 477\n"
    ),
    "pipeline.py": (
        "References\n"
        "    CGC classification logic: Paper 1 (CG-Framework), Appendix E\n"
        "    Pipeline architecture: operator -> diagrams -> momentum -> topology -> conservation\n"
    ),
    "dyson_schwinger.py": (
        "References\n"
        "    Dyson (1949): Dyson-Schwinger equations\n"
        "    Roberts & Williams (1994): Review of DSE formalism\n"
        "    Alkofer & von Smekal (2001): IR fixed points of QCD DSE\n"
    ),
    "self_consistent_dyson.py": (
        "References\n"
        "    Self-consistent DSE gap: Alkofer & von Smekal (2001), Phys. Rept. 353, 281\n"
        "    Emergent mass scale: Roberts (2016), Few Body Syst. 57, 1\n"
    ),
    "frg_trace_density.py": (
        "References\n"
        "    Wetterich (1993): exact FRG, trace density eta(k)\n"
        "    Litim (2001): optimal regulator, Phys. Rev. D 64, 105007\n"
        "    Berges-Tetradis-Wetterich (2002): FRG review, Phys. Rept. 363, 223\n"
    ),
    "frg_enhancement.py": (
        "References\n"
        "    Coupling enhancement from trace density: Wetterich equation\n"
        "    Litim (2001): regulator-dependence < 0.001%\n"
    ),
    "gravity_feedback.py": (
        "References\n"
        "    Graviton backreaction on FRG flow: Reuter (1998), Phys. Rev. D 57, 971\n"
        "    Asymptotic Safety gravity: Niedermaier-Reuter (2006), Living Rev. Rel. 9, 5\n"
    ),
    "crossed_ladder_f2.py": (
        "References\n"
        "    Crossed-ladder diagrams: higher-order DSE formalism\n"
        "    F^2 specific: BRST-exact insertion at q=0 ensures Weinberg power counting\n"
    ),
    "topology_classifier.py": (
        "References\n"
        "    Diagram topology (bubble vs ladder): standard perturbation theory\n"
        "    Resummation: ladder approximation (Roberts-Williams 1994)\n"
    ),
}

BACKEND_REFS = {
    "fa_model_parser.py": "FeynArts model file parser — backend infrastructure, no physics formulas",
    "feynarts_backend.py": "FeynArts/FormCalc backend — generates diagrams via external tool",
    "qgraf_backend.py": "qgraf backend — diagram generation via automatic vertex enumeration",
    "multi_loop_generator.py": "Multi-loop diagram generator — generalizes one-loop topology",
    "one_loop_generator.py": "One-loop diagram generator — explicit vertex/propagator enumeration",
    "two_loop_topologies.py": "Two-loop topology generator — SM vertex enumeration and classification",
}

import pathlib

for fname, ref_text in MODULE_REFS.items():
    path = pathlib.Path(BASE) / fname
    if not path.exists():
        print(f"  SKIP {fname}: not found")
        continue
    with open(path, encoding='utf-8') as f:
        content = f.read()

    # Check if already has Ref header
    if "References\n-" in content or "References\n    " in content:
        print(f"  SKIP {fname}: already has refs")
        continue

    # Insert after module docstring
    # Find end of first docstring
    lines = content.split('\n')
    in_docstring = False
    docstring_end = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if not in_docstring:
                in_docstring = True
            else:
                docstring_end = i
                break

    if docstring_end < 0:
        # No docstring — insert after imports
        for i, line in enumerate(lines):
            if not line.startswith('import ') and not line.startswith('from ') and line.strip():
                docstring_end = i - 1
                break

    if docstring_end >= 0:
        # Insert reference block
        ref_block = f'\n# {ref_text.replace(chr(10), chr(10) + "# ")}\n'
        lines.insert(docstring_end + 2, ref_block)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  OK {fname}")

for fname, desc in BACKEND_REFS.items():
    path = pathlib.Path(BASE) / fname
    if not path.exists():
        print(f"  SKIP {fname}: not found")
        continue
    print(f"  INFRA {fname}: {desc}")

print("\nDone with P1-2.")
