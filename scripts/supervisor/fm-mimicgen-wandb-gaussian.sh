#!/usr/bin/env bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
set -e
cd /workspace/fm_dno
exec /venv/fm_dno/bin/wandb beta sync --live --yes --no-skip-synced --entity andyliu7081-northeastern-university --project fm_dno_mimicgen /workspace/fm_dno/output/mimicgen6/headinggaussian_dit_seed42/wandb/latest-run
