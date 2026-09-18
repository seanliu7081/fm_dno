#!/usr/bin/env bash
set -eo pipefail
. /opt/supervisor-scripts/utils/logging.sh /workspace/fm_dno/output/shared_holdout_heading_smq_20260914/logs/plotting.log
. /opt/supervisor-scripts/utils/environment.sh
export PYTHONPATH=/workspace/fm_dno
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
cd /workspace/fm_dno
pty /venv/fm_dno/bin/python scripts/plot_shared_holdout_comparison.py --output output/shared_holdout_heading_smq_20260914 2>&1
