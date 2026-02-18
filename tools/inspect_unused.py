#!/usr/bin/env python3
import ast
import os
from collections import defaultdict

ROOT = "."  # run from project root
SKIP_DIRS = {".venv", "__pycache__", ".git", "node_modules"}

def iter_py_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)

def module_name_from_path(path):
    p = os.path.relpath(path, ROOT).replace(os.sep, ".")
    return p[:-3] if p.endswith(".py") else p

def main():
    defined = defaultdict(set)   # module -> {func names}
    called = defaultdict(set)    # module -> {called names (simple)}
    imported = defaultdict(set)  # module -> {imported names}

    for path in iter_py_files(ROOT):
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()

        mod = module_name_from_path(path)

        try:
            tree = ast.parse(src, filename=path)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined[mod].add(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                defined[mod].add(node.name)
            elif isinstance(node, ast.ImportFrom):
                for n in node.names:
                    imported[mod].add(n.name)
            elif isinstance(node, ast.Call):
                # simple calls: foo(...)
                if isinstance(node.func, ast.Name):
                    called[mod].add(node.func.id)
                # attr calls: x.foo(...)
                elif isinstance(node.func, ast.Attribute):
                    called[mod].add(node.func.attr)

    # global called set
    called_all = set()
    for m in called:
        called_all |= called[m]

    print("\n=== Possibly unused functions (defined but never called by name) ===\n")
    total = 0
    for m in sorted(defined):
        unused = sorted([fn for fn in defined[m] if fn not in called_all and not fn.startswith("_")])
        if unused:
            print(f"\n[{m}]")
            for fn in unused:
                print("  -", fn)
                total += 1

    print(f"\nTotal candidates: {total}")
    print("\nNOTE: This is heuristic. Streamlit/dynamic usage can hide calls.\n")

if __name__ == "__main__":
    main()