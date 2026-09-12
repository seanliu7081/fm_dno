#!/usr/bin/env bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
export WANDB_MODE=online WANDB_RUN_ID=cc5kl9wl WANDB_ENTITY=andyliu7081-northeastern-university
cd /workspace/fm_dno
pty /workspace/fm_dno/scripts/run_mimicgen_training.sh gaussian 2,3 2>&1
