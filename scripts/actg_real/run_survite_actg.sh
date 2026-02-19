#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="./data"
RESULT_DIR="./results"

NUM_REPEATS=10
TRAIN_SIZE=0.5
VAL_SIZE=0.25
TEST_SIZE=0.25

DATASETS=("actgLC" "actgHC")

# Hyperparams
IPM_TYPE="wasserstein"
BETA=0.001
EPOCHS=1500
BATCH_SIZE=256

for DATASET in "${DATASETS[@]}"; do
  echo "============================================================"
  echo "Running SurvITE | dataset=${DATASET} | train=${TRAIN_SIZE} val=${VAL_SIZE} test=${TEST_SIZE}"
  echo "============================================================"

  python benchmark/run_survite.py \
    --num_repeats "$NUM_REPEATS" \
    --dataset_name "$DATASET" \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --train_size "$TRAIN_SIZE" \
    --val_size "$VAL_SIZE" \
    --test_size "$TEST_SIZE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --ipm_type "$IPM_TYPE" \
    --beta "$BETA" \
    --verbose
done

echo "All runs completed."