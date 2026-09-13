#!/usr/bin/env python3
"""Model-side entry point; run in the full StarVLA training virtualenv."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oat.starvla_heading.server import main

if __name__ == "__main__":
    raise SystemExit(main())
