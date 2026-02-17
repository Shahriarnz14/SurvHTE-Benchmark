#!/usr/bin/env bash
set -euo pipefail


DATA_DIR="./data"
RESULT_DIR="./results"
DATASET="synthetic"
NUM_REPEATS=10
# TRAIN_SIZE=5000 # (vary this in the loop below)
VAL_SIZE=2500
TEST_SIZE=2500

# Hyperparams 
IPM_TYPE="wasserstein"
BETA=0.001
EPOCHS=1500
BATCH_SIZE=256

TRAIN_SIZES=(50 100 200 300 500 1000 2000 5000 10000)

for TS in "${TRAIN_SIZES[@]}"; do
  if [[ "$TS" -eq 5000 ]]; then
    echo "==> Skipping train_size=5000 (already ran)."
    continue
  fi

  echo "============================================================"
  echo "Running SurvITE | dataset=${DATASET} | train_size=${TS}"
  echo "============================================================"

  python benchmark/run_survite.py \
    --num_repeats "$NUM_REPEATS" \
    --dataset_name "$DATASET" \
    --data_dir ${DATA_DIR} \
    --result_dir ${RESULT_DIR} \
    --train_size "$TS" \
    --val_size "$VAL_SIZE" \
    --test_size "$TEST_SIZE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --ipm_type "$IPM_TYPE" \
    --beta "$BETA" \
    --verbose
done

echo "All runs completed."