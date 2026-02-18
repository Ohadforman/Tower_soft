#!/usr/bin/env bash
set -euo pipefail

echo "== Ruff: unused imports/vars =="
ruff check . --select F401,F841 || true

echo
echo "== Vulture: unused code candidates =="
vulture . --min-confidence 70 || true

echo
echo "== AST heuristic unused functions =="
python tools/inspect_unused.py || true