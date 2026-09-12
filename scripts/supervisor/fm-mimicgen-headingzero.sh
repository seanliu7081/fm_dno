#!/usr/bin/env bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
export WANDB_MODE=online WANDB_RUN_ID=1hg9rqh5 WANDB_ENTITY=andyliu7081-northeastern-university
cd /workspace/fm_dno
pty /workspace/fm_dno/scripts/run_mimicgen_training.sh zero 0,1 2>&1
