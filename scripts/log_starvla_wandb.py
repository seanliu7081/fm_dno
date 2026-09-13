#!/usr/bin/env python3
"""Stream existing StarVLA experiment metrics to online W&B."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oat.starvla_heading.wandb_logger import main

if __name__ == "__main__":
    main()
