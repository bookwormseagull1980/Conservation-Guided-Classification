"""Batch mypy fix: add missing return type annotations and float() casts."""
import re, sys, pathlib

BASE = pathlib.Path(r"D:\论文撰写\Conservation-Guided Classification\cgc\engine")

# Files to skip (manual fixes needed)
SKIP = set()

# Pattern 1: def ...(self, ...): with no return type and docstring starting with """
# For methods that clearly return None (update, set, build patterns)
def add_return_none(content: str, method_names: list[str]) -> str:
    for name in method_names:
        pattern = rf'(def {name}\(self,.*?\):)\s*\n(\s+)"""'
        content = re.sub(pattern, rf'\1 -> None:\n\2"""', content)
    return content

# Pattern 2: np.exp/op expressions that return Any
def wrap_float_returns(content: str) -> str:
    # Find patterns like "return k * k * np.exp(...)" or "return np.xxx(...)"
    pattern = r'(\s{8}return )((?:k|lam|m2|V)\s*[*/+\-]?\s*)+(?:np\.\w+\([^)]+\))(?!.*\bfloat\()'
    # Too complex for regex; do file-specific fixes
    return content


# Fix files individually
fixes = {
    "diagram_builder.py": {
        # Add -> None to void methods
        "no_untyped": [
            ("def _init_graph(", "def _init_graph(self, n_vertices: int) -> None:"),
        ],
    },
}


for filepath, fix_info in fixes.items():
    full = BASE / filepath
    if not full.exists():
        continue
    content = full.read_text(encoding='utf-8')
    modified = False

    for category, patches in fix_info.items():
        for old, new in patches:
            if old in content:
                content = content.replace(old, new)
                modified = True
                print(f"  {filepath}: {old[:40]}... -> {new[:40]}...")

    if modified:
        full.write_text(content, encoding='utf-8')
        print(f"  Wrote {filepath}")

print("Done.")
