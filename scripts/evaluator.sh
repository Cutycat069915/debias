#!/bin/bash

set -e

PYTHON=python
SCRIPT=./Evaluators/evaluate.py

TEXT_ROOT=./text_save
SPLIT=./split_dataset
RESULT_ROOT=./Evaluators/text_results

mkdir -p "$RESULT_ROOT"

for TESTCASE_DIR in "$TEXT_ROOT"/R16_*; do
    [ -d "$TESTCASE_DIR" ] || continue

    TESTCASE=$(basename "$TESTCASE_DIR")

    echo "========================================"
    echo "Evaluating $TESTCASE"
    echo "========================================"

    ADAPTERS=("$TESTCASE_DIR"/seed_*)

    OUT_DIR="$RESULT_ROOT/$TESTCASE"
    OUT="$OUT_DIR/results.json"

    mkdir -p "$OUT_DIR"

    $PYTHON "$SCRIPT" \
        "${ADAPTERS[@]}" \
        --split "$SPLIT" \
        --out "$OUT"
done
