#!/usr/bin/env python3
import os
import runpy
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TARGET = os.path.join(ROOT_DIR, "scripts", "cli", "run_preflight.py")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if __name__ == "__main__":
    runpy.run_path(TARGET, run_name="__main__")
