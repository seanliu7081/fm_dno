#!/usr/bin/env python3
"""Live terminal progress for the StarVLA evaluation summaries (read-only)."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time


def render(root, seeds):
    width = min(36, max(12, shutil.get_terminal_size((100, 24)).columns - 65))
    lines = ['Heading-Gaussian StarVLA evaluation', time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), '']
    for seed in seeds:
        lines.append(f'Seed {seed}')
        for benchmark, title, expected in [('libero', 'LIBERO', 2000), ('libero_plus', 'LIBERO-Plus', 10030)]:
            path = root / f'seed{seed}' / 'evaluation' / benchmark / 'summary.json'
            try:
                summary = json.loads(path.read_text())
            except FileNotFoundError:
                summary = None
            except (OSError, json.JSONDecodeError):
                lines.append(f'  {title:<11} Summary temporarily unavailable')
                continue
            overall = summary.get('overall', {}) if summary else {}
            done = overall.get('completed', 0)
            total = overall.get('expected', expected)
            errors = overall.get('errors', 0)
            fraction = min(1.0, done / total) if total else 0.0
            filled = int(width * fraction)
            bar = '#' * filled + '-' * (width - filled)
            status = 'complete' if summary and summary.get('complete') else ('saved progress' if summary else 'waiting')
            lines.append(f'  {title:<11} [{bar}] {fraction:6.2%}  {done:>5}/{total:<5}  {status}')
            if errors:
                lines.append(f'                Episode errors: {errors}')
            if summary and summary.get('complete') and summary.get('official_success_rate') is not None:
                lines.append(f'                Final success rate: {summary["official_success_rate"]:.2%}')
        lines.append('')
    lines.append('Refresh: 2 seconds | Ctrl+C to exit | Progress comes from saved summaries.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'output/starvla_heading_gaussian')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    live = sys.stdout.isatty() and not args.once
    try:
        while True:
            if live:
                print('\033[2J\033[H', end='')
            print(render(args.root, args.seeds), flush=True)
            if not live:
                return
            time.sleep(2)
    except KeyboardInterrupt:
        print('\nMonitor closed; evaluation continues.')


if __name__ == '__main__':
    main()
