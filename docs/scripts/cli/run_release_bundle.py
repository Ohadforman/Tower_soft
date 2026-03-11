#!/usr/bin/env python3
import os
import runpy
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TARGET = os.path.join(ROOT_DIR, "scripts", "cli", "run_release_bundle.py")

if __name__ == "__main__":
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    runpy.run_path(TARGET, run_name="__main__")

