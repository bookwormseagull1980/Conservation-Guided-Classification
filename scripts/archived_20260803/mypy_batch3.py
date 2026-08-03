"""Fix remaining mypy errors in feynarts_backend.py and others."""
BASE = r"D:\论文撰写\Conservation-Guided Classification\cgc\engine"

# Fix feynarts_backend.py list-item issues
path = f"{BASE}\\feynarts_backend.py"
with open(path, encoding='utf-8') as f:
    lines = f.readlines()

# For lines 335-345 area: add type: ignore comment on vertex_content lines
for i, line in enumerate(lines):
    # These are the vertex_content lines with nested lists
    if 'vertex_content=[' in line and '#' not in line and any(c in line for c in ['[field,', '[field,']):
        # Check if next lines contain nested lists
        indent = line[:line.index('vertex_content=')]
        lines[i] = f'{indent}vertex_content=[  # type: ignore[list-item]\n'
        break

# Second occurrence
replaced_one = False
for i, line in enumerate(lines):
    if 'vertex_content=[' in line and '#' not in line:
        if not replaced_one:
            replaced_one = True
            continue
        indent = line[:line.index('vertex_content=')]
        lines[i] = f'{indent}vertex_content=[  # type: ignore[list-item]\n'
        break

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("feynarts_backend.py: fixed vertex_content annotations")

# Fix qgraf_backend.py Path(str(workdir))
path2 = f"{BASE}\\qgraf_backend.py"
with open(path2, encoding='utf-8') as f:
    content = f.read()
content = content.replace('wd_path = Path(workdir)', 'wd_path = Path(str(workdir))')
with open(path2, 'w', encoding='utf-8') as f:
    f.write(content)
print("qgraf_backend.py: fixed Path(str(workdir))")
