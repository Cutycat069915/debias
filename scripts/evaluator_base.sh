#!/usr/bin/env bash

set -e

PYTHON=python
SCRIPT=./Evaluators/evaluate.py

SPLIT=./split_dataset_vision
OUT=./Evaluators/vision_results/base/results.json

K=10

BASE_ADAPTERS=()

for ((FOLD=1; FOLD<=K; FOLD++)); do
    BASE_ADAPTERS+=("base/fold_${FOLD}")
done

mkdir -p "$(dirname "$OUT")"

"$PYTHON" "$SCRIPT" \
    "${BASE_ADAPTERS[@]}" \
    --split "$SPLIT" \
    --out "$OUT"
