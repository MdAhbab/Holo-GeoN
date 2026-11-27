#!/usr/bin/env bash
# remote/colab_run.sh - run on the Colab A100 VM (from `colab console -s S`), after Drive is mounted.
#
# Drive layout it expects (copy once from your PC; the folders are yours to create):
#   $DRIVE/code/Holo-GeoN/            this repository
#   $DRIVE/data/megascale_ddg_pairs.parquet
#   $DRIVE/data/splits_megascale.parquet
#   $DRIVE/data/fireprotdb_holdout.parquet
# Results and the resume packs go to $DRIVE/runs/<manifest name>, so a restarted VM
# restores finished items and continues.
#
#   bash remote/colab_run.sh plan        # cost plan only, no GPU needed
#   MAX_HOURS=1.5 bash remote/colab_run.sh run     # first wave: measures s/step
#   MAX_HOURS=6   bash remote/colab_run.sh run     # after re-planning
set -euo pipefail

DRIVE="${DRIVE:-/content/drive/MyDrive/holo_geon}"
MANIFEST="${MANIFEST:-configs/experiments/colab_a100_benchmark.yaml}"
NAME="$(basename "$MANIFEST" .yaml)"
MAX_HOURS="${MAX_HOURS:-1.5}"
REPO=/content/Holo-GeoN
MODE="${1:-plan}"

if [ ! -d "$REPO" ]; then
  cp -r "$DRIVE/code/Holo-GeoN" "$REPO"
fi
mkdir -p /content/data
for f in megascale_ddg_pairs.parquet splits_megascale.parquet fireprotdb_holdout.parquet; do
  [ -f "/content/data/$f" ] || cp "$DRIVE/data/$f" /content/data/
done

# transformers must stay below 5 (ESM contact extraction); torch is the image's own.
pip install -q 'transformers>=4.40,<5' torch_geometric ViennaRNA pyyaml pyarrow \
    scikit-learn scipy pandas tqdm

cd "$REPO"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -m scripts.remote_job --manifest "$MANIFEST" --plan

if [ "$MODE" = "run" ]; then
  # Foreground on purpose when called through `colab console`; start it under
  # nohup there if you want to close the console:
  #   nohup bash remote/colab_run.sh run > /content/job.log 2>&1 &
  python -m scripts.remote_job --manifest "$MANIFEST" \
      --persist "$DRIVE/runs/$NAME" --out-root "/content/holo/runs/$NAME" \
      --data-root /content/data --max-hours "$MAX_HOURS"
fi
