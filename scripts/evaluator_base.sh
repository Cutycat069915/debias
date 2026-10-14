#!/usr/bin/env bash

set -e

PYTHON=python
SCRIPT=./Evaluators/evaluate.py

SPLIT=./split_dataset
OUT=./Evaluators/text_results/base/results.json

SEEDS=(15 22 23 32 432)

BASE_ADAPTERS=()

for SEED in "${SEEDS[@]}"; do
    BASE_ADAPTERS+=("base/seed_${SEED}")
done

mkdir -p "$(dirname "$OUT")"

$PYTHON "$SCRIPT" \
    "${BASE_ADAPTERS[@]}" \
    --split "$SPLIT" \
    --out "$OUT"
