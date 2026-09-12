#!/usr/bin/env bash
# Add pinned MimicGen task code to the existing fm_dno Conda environment.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate fm_dno
for specification in \
  'mimicgen https://github.com/NVlabs/mimicgen.git 72bd767c255545f462e7ccfb2731f2e5d4c1d9bb' \
  'robosuite-task-zoo https://github.com/ARISE-Initiative/robosuite-task-zoo.git 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed'; do
  read -r name url revision <<< "$specification"
  if [[ ! -d "third_party/$name" ]]; then
    git clone "$url" "third_party/$name"
    git -C "third_party/$name" checkout "$revision"
  elif [[ "$(git -C "third_party/$name" rev-parse HEAD)" != "$revision" ]]; then
    echo "third_party/$name is not at the required revision $revision" >&2
    exit 1
  fi
done
# The official installation guide recommends dropping task-zoo's deprecated
# mujoco-py requirement. These environments use the installed mujoco bindings.
python - <<'PY'
from pathlib import Path
p=Path('third_party/robosuite-task-zoo/setup.py')
p.write_text(p.read_text().replace('        "mujoco-py>=2.0.2.9",\n',''))
PY
python -m pip install --no-deps -e third_party/robosuite-task-zoo -e third_party/mimicgen
python -m pip install --no-deps robosuite==1.4.1 mujoco==3.2.6
python -m pip install chardet pynput
python -m pip install --no-deps -e .
python -m pip check
