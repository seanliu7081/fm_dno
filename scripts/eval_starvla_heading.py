#!/usr/bin/env python3
"""Simulator-side entry point; use an isolated LIBERO or LIBERO-Plus virtualenv."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oat.starvla_heading.evaluation import main

if __name__ == "__main__":
    raise SystemExit(main())
