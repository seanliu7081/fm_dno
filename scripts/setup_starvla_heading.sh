#!/usr/bin/env bash
# Set up pinned Qwen + StarVLA training dependencies and optional benchmark assets.
# Usage: bash scripts/setup_starvla_heading.sh [--download] [--with-plus]
# Reuses an existing CUDA-enabled Python environment without modifying it.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
setup_root=${STARVLA_WORKSPACE:-/workspace}
base_python=${STARVLA_BASE_PYTHON:-/venv/fm_dno/bin/python}
training_env=${STARVLA_TRAIN_ENV:-${setup_root}/venvs/starvla-heading}
plus_env=${LIBERO_PLUS_ENV:-${setup_root}/venvs/libero-plus}
download=0
with_plus=0
for arg in "$@"; do
    case "$arg" in
        --download) download=1 ;;
        --with-plus) with_plus=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

"$base_python" -c 'import torch, torchvision; assert torch.cuda.is_available(), "CUDA-enabled torch is required"; print(torch.__version__, torchvision.__version__)'
mkdir -p "$setup_root/venvs" "$setup_root/deps" "$setup_root/models"
if [[ ! -x "$training_env/bin/python" ]]; then
    "$base_python" -m venv --system-site-packages "$training_env"
fi
DS_BUILD_OPS=0 "$training_env/bin/pip" install --no-build-isolation -r "$repo_root/requirements-starvla-heading.txt"

starvla_path="$setup_root/deps/starVLA"
starvla_revision=3422b9f2387b6f682cf02802904a77b23ab13afd
if [[ ! -d "$starvla_path/.git" ]]; then
    git clone https://github.com/starVLA/starVLA.git "$starvla_path"
    git -C "$starvla_path" checkout "$starvla_revision"
fi
[[ $(git -C "$starvla_path" rev-parse HEAD) == "$starvla_revision" ]] || {
    echo "Existing StarVLA checkout must be at $starvla_revision: $starvla_path" >&2
    exit 1
}
"$training_env/bin/pip" install --no-deps --editable "$starvla_path"

# Keep simulator package selection and runtime configuration explicit.
"$training_env/bin/python" - "$setup_root" "$repo_root" <<'PYCONFIG'
import pathlib
import sys
import yaml
workspace, repo = map(pathlib.Path, sys.argv[1:])
root = repo / 'third_party/LIBERO/libero/libero'
config = workspace / 'libero_config/original'
config.mkdir(parents=True, exist_ok=True)
(config / 'config.yaml').write_text(yaml.safe_dump({
    'benchmark_root': str(root), 'bddl_files': str(root / 'bddl_files'),
    'init_states': str(root / 'init_files'), 'assets': str(root / 'assets'),
    'datasets': str(workspace / 'shared_data/libero_hdf5'),
}))
PYCONFIG

if (( download )); then
    # Download only the original four-suite demonstrations; Plus remains evaluation-only.
    HF_HUB_DISABLE_XET=1 "$training_env/bin/python" - "$setup_root" <<'PY'
import fnmatch
import hashlib
import json
import pathlib
import sys
from huggingface_hub import HfApi, snapshot_download

root = pathlib.Path(sys.argv[1])
jobs = [
    ('Qwen/Qwen2.5-VL-3B-Instruct', 'model', '66285546d2b821cf421d4f5eb2576359d3770cd3',
     root / 'models/Qwen2.5-VL-3B-Instruct', ['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja']),
    ('yifengzhu-hf/LIBERO-datasets', 'dataset', 'f13aa24a3da8c43c7225569f28c562979fa0e35a',
     root / 'shared_data/libero_hdf5',
     ['libero_spatial/*.hdf5', 'libero_object/*.hdf5', 'libero_goal/*.hdf5', 'libero_10/*.hdf5']),
]
for repo, kind, revision, dest, patterns in jobs:
    snapshot_download(repo, repo_type=kind, revision=revision, local_dir=dest,
                      allow_patterns=patterns, max_workers=4)
    files = []
    for entry in HfApi().repo_info(repo, repo_type=kind, revision=revision, files_metadata=True).siblings:
        if not any(fnmatch.fnmatchcase(entry.rfilename, pattern) for pattern in patterns):
            continue
        path = dest / entry.rfilename
        assert path.is_file() and path.stat().st_size == entry.size, str(path)
        if entry.lfs:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                while chunk := stream.read(16 * 1024 * 1024):
                    digest.update(chunk)
            assert digest.hexdigest() == entry.lfs.sha256, str(path)
        files.append({'path': entry.rfilename, 'bytes': entry.size,
                      'sha256': entry.lfs.sha256 if entry.lfs else None})
    (dest / 'SOURCE_MANIFEST.json').write_text(json.dumps({
        'repo_id': repo, 'repo_type': kind, 'revision': revision,
        'files': files, 'verified_sizes': True, 'verified_lfs_sha256': True,
    }, indent=2) + '\n')
    print(f'Ready: {dest} ({len(files)} files)')
PY
fi

if (( with_plus )); then
    plus_path="$setup_root/deps/LIBERO-plus"
    plus_revision=4976dc30028e805ff8094b55501d532c48fec182
    if [[ ! -d "$plus_path/.git" ]]; then
        git clone https://github.com/sylvestf/LIBERO-plus.git "$plus_path"
        git -C "$plus_path" checkout "$plus_revision"
    fi
    [[ $(git -C "$plus_path" rev-parse HEAD) == "$plus_revision" ]] || {
        echo "Existing LIBERO-Plus checkout must be at $plus_revision: $plus_path" >&2
        exit 1
    }
    if [[ ! -x "$plus_env/bin/python" ]]; then
        "$base_python" -m venv --system-site-packages "$plus_env"
    fi
    # The base environment supplies robosuite==1.4.1 and mujoco==3.2.6.
    "$plus_env/bin/pip" install --no-deps --editable "$plus_path"
    # Official fog corruption uses np.float_, which was removed in NumPy 2.
    "$plus_env/bin/pip" install 'numpy==1.26.4' 'opencv-python==4.10.0.84' 'wand==0.6.13' 'scikit-image==0.25.2'
    "$plus_env/bin/python" - "$plus_env" "$plus_path" "$setup_root" <<'PY'
import pathlib
import sys
import sysconfig
import yaml
env, repo, workspace = map(pathlib.Path, sys.argv[1:])
root = repo / 'libero/libero'
config = workspace / 'libero_config/plus'
config.mkdir(parents=True, exist_ok=True)
content = yaml.safe_dump({
    'benchmark_root': str(root), 'bddl_files': str(root / 'bddl_files'),
    'init_states': str(root / 'init_files'), 'assets': str(root / 'assets'),
    'datasets': str(workspace / 'shared_data/libero_hdf5'),
})
(config / 'config.yaml').write_text(content)
legacy = env / 'libero_config'
legacy.mkdir(exist_ok=True)
(legacy / 'config.yaml').write_text(content)
# An inherited original LIBERO editable install must not win import resolution.
(pathlib.Path(sysconfig.get_path('purelib')) / '000_libero_plus.pth').write_text(
    f'import sys; sys.path.insert(0, {str(repo)!r})\n')
PY
    if (( download )); then
        HF_HUB_DISABLE_XET=1 "$training_env/bin/python" - "$setup_root" "$plus_path" <<'PY'
import hashlib
import json
import pathlib
import sys
import zipfile
from huggingface_hub import hf_hub_download
workspace, repo = map(pathlib.Path, sys.argv[1:])
revision = 'dd2bd61b7d9a6fef1abc52d606e983b41886a149'
archive = pathlib.Path(hf_hub_download(
    'Sylvest/LIBERO-plus', repo_type='dataset', filename='assets.zip', revision=revision,
    local_dir=workspace / 'shared_data/libero_plus_assets'))
digest = hashlib.sha256()
with archive.open('rb') as stream:
    while chunk := stream.read(16 * 1024 * 1024):
        digest.update(chunk)
checksum = digest.hexdigest()
assert checksum == '96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf'
dest = (repo / 'libero/libero').resolve()
# The pinned official archive includes its author's absolute host directory
# as a relative prefix, so ordinary extractall does not produce assets/.
import shutil
prefix = pathlib.PurePosixPath('inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets')
count = 0
with zipfile.ZipFile(archive) as archive_file:
    for info in archive_file.infolist():
        name = pathlib.PurePosixPath(info.filename)
        assert not name.is_absolute() and '..' not in name.parts, name
        if not name.is_relative_to(prefix):
            assert info.is_dir(), name
            continue
        target = dest / 'assets' / name.relative_to(prefix)
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive_file.open(info) as source, target.open('wb') as output:
            shutil.copyfileobj(source, output)
        count += 1
(dest / 'ASSETS_SOURCE_MANIFEST.json').write_text(json.dumps({
    'repo_id': 'Sylvest/LIBERO-plus', 'revision': revision, 'asset': 'assets.zip',
    'sha256': checksum, 'sha256_verified': True, 'extracted_to': str(dest / 'assets'),
    'archive_prefix': str(prefix), 'file_count': count,
}, indent=2) + '\n')
PY
    fi
    # On Ubuntu, install libmagickwand-dev if this fails. Required for sensor-noise tasks.
    "$plus_env/bin/python" -c 'import wand.image, skimage'
    echo "Plus interpreter: $plus_env/bin/python"
    echo "Plus config: export LIBERO_CONFIG_PATH=$setup_root/libero_config/plus"
fi

"$training_env/bin/python" -c 'import deepspeed, qwen_vl_utils; from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead; print("Training imports OK")'
echo "Training interpreter: $training_env/bin/python"
