# -*- coding: utf-8 -*-
"""Fix imports: cgc.engine.<rp3_module> -> cgc.rp3_engine.<rp3_module> (2026-08-03)."""
import io
import os
import re

ROOT = r'D:\论文撰写\Conservation-Guided Classification'

RP3_MODULES = [
    "frg_flow", "frg_flow_rp3", "frg_trace_density", "self_consistent_dyson",
    "gravity_feedback", "frg_enhancement", "crossed_ladder_f2",
    "dyson_schwinger", "chi_potential",
]

# Patterns to fix:
# 1. from cgc.engine.<mod> import ...  ->  from cgc.rp3_engine.<mod> import ...
# 2. import cgc.engine.<mod>           ->  import cgc.rp3_engine.<mod>
# 3. cgc.engine.<mod>.xxx              ->  cgc.rp3_engine.<mod>.xxx

pat_abs = re.compile(r'cgc\.engine\.(' + '|'.join(RP3_MODULES) + r')')
pat_rel_engine = re.compile(r'from \.(' + '|'.join(RP3_MODULES) + r') import')

changed_files = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    if '__pycache__' in dirpath or '.git' in dirpath:
        continue
    for fn in filenames:
        if not fn.endswith('.py'):
            continue
        path = os.path.join(dirpath, fn)
        with io.open(path, encoding='utf-8', newline='') as f:
            content = f.read()
        new = content
        # absolute: cgc.engine.X -> cgc.rp3_engine.X
        new = pat_abs.sub(lambda m: 'cgc.rp3_engine.' + m.group(1), new)
        # relative from .X import -> from .rp3_engine.X import (only in cgc/engine/)
        if dirpath.endswith('cgc' + os.sep + 'engine'):
            new = pat_rel_engine.sub(lambda m: 'from .rp3_engine.' + m.group(1) + ' import', new)
        if new != content:
            with io.open(path, 'w', encoding='utf-8', newline='') as f:
                f.write(new)
            changed_files.append(path)
            print(f"fixed: {os.path.relpath(path, ROOT)}")

print(f"\n{len(changed_files)} files updated")
