#!/usr/bin/env bash
# remote/colab_l4_ui_recapture.sh - recapture the seven UI screenshots (ui_01..ui_07, 2x DPI)
# on a Colab L4 from the A100 release bundle on Drive. No training; the interface makes the
# same requests a user would. Run it from a notebook cell AFTER Drive is mounted:
#
#   from google.colab import drive; drive.mount('/content/drive')
#   !git clone -q -b version-2 https://github.com/MdAhbab/Holo-GeoN /content/Holo-GeoN 2>/dev/null || git -C /content/Holo-GeoN pull -q --ff-only
#   !bash /content/Holo-GeoN/remote/colab_l4_ui_recapture.sh
#
# Outputs (mirrored to Drive as each file is written):
#   $OUT/ui_01_ddg_ubq_L8P.png ... ui_07_compare_modes_3d.png, $OUT/panels/*.png,
#   $OUT/ui_recapture_report.json (versions, health snapshots, per-shot checks), uvicorn logs.
#
# Cost: about 6-8 min wall clock on a fresh L4 VM (installs + ESM-2 650M download dominate;
# the captures take about 2 min), i.e. roughly 0.15-0.2 compute units at ~1.5 units/h.
set -euo pipefail

REPO="${REPO:-/content/Holo-GeoN}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/holo_geon_g4}"
BUNDLE="${BUNDLE:-$DRIVE_ROOT/runs/a100_suite/block3_demo/release_bundle}"
OUT="${OUT:-$DRIVE_ROOT/runs/a100_suite/ui_recapture_l4}"

export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export HF_HOME="${HF_HOME:-/content/hf_cache}"

if [ ! -d "$BUNDLE" ]; then
  echo "release bundle not found: $BUNDLE (mount Drive first: from google.colab import drive; drive.mount('/content/drive'))" >&2
  exit 2
fi
cd "$REPO"
echo "[l4-ui] repo $(git rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git rev-parse --short HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[l4-ui] WARNING: no GPU visible"

# transformers must stay below 5 (ESM-2 contact extraction needs eager attention); torch is the image's own.
pip install -q 'transformers>=4.40,<5' torch_geometric ViennaRNA fastapi 'uvicorn[standard]' requests pydantic playwright
python -m playwright install --with-deps chromium > /tmp/playwright_install.log 2>&1 \
  || python -m playwright install chromium >> /tmp/playwright_install.log 2>&1

python -m remote.gpu_jobs.l4_ui_recapture \
  --bundle "$BUNDLE" --out "$OUT" --drive-root "$DRIVE_ROOT" --scale 2 --panels "$@"
echo "[l4-ui] done -> $OUT"
ls -la "$OUT"
