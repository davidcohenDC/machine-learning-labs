#!/usr/bin/env bash
# Executes a notebook top to bottom, in its own folder, and writes the run
# next to it as <name>.run.ipynb so the committed file is left alone.
#
#   scripts/run.sh 01-svm-knn/svm-knn.ipynb
#   scripts/run.sh 03-solar-regression/regression.ipynb 1800   (timeout per cell, seconds)
set -euo pipefail

nb=${1:?usage: scripts/run.sh <folder/notebook.ipynb> [cell-timeout]}
timeout=${2:-3600}
cd "$(dirname "$nb")"
name=$(basename "$nb" .ipynb)
jupyter nbconvert --to notebook --execute "$name.ipynb" \
    --ExecutePreprocessor.timeout="$timeout" --output "$name.run.ipynb"
echo "ok: $(dirname "$nb")/$name.run.ipynb"
