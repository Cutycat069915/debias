#!/bin/bash
set -e 

PYTHON=python
SCRIPT=./Evaluators/evaluate.py

VISION_ROOT=./text_save
SPLIT=./split_dataset_vision
RESULT_ROOT=./Evaluators/textv_results

mkdir -p "$RESULT_ROOT"

for TESTCASE_DIR in "$VISION_ROOT"/R16_*; do
    [ -d "$TESTCASE_DIR" ] || continue

    TESTCASE=$(basename "$TESTCASE_DIR")

    echo "========================================"
    echo "Evaluating $TESTCASE"
    echo "========================================"

    ADAPTERS=("$TESTCASE_DIR"/fold_*)

    OUT_DIR="$RESULT_ROOT/$TESTCASE"
    OUT="$OUT_DIR/results.json"

    mkdir -p "$OUT_DIR"

    $PYTHON "$SCRIPT" \
        "${ADAPTERS[@]}" \
        --split "$SPLIT" \
        --out "$OUT"
done
