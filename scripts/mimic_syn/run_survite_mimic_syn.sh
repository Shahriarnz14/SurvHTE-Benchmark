#!/usr/bin/env bash
set -euo pipefail

# Paths / basic settings
DATA_DIR="./data"
RESULT_DIR="./results"
MODEL_DIR="./models"

DATASET="mimic_syn"
NUM_REPEATS=10
TRAIN_SIZE=0.5
VAL_SIZE=0.25
TEST_SIZE=0.25


# Hyperparams
IPM_TYPE="wasserstein"
BETA=0.001
EPOCHS=5000
BATCH_SIZE=256

DIMs=(16 32 64 128)
LAYERs=(2) # extend, e.g., (2 3)


# Logging
LOG_DIR="./results/survite/logs/${DATASET}"
mkdir -p "$LOG_DIR"
timestamp() { date +"%Y%m%d_%H%M%S"; }

for DIM in "${DIMs[@]}"; do
  for LAYER in "${LAYERs[@]}"; do
    EXP_NAME="dim-${DIM}_layer-${LAYER}"
    TS="$(timestamp)"
    LOG_FILE="${LOG_DIR}/${EXP_NAME}_${TS}.log"

    echo "============================================================"
    echo "Running SurvITE | dataset=${DATASET} | ${EXP_NAME}"
    echo "Log: ${LOG_FILE}"
    echo "============================================================"

    python benchmark/run_survite.py \
      --num_repeats "$NUM_REPEATS" \
      --dataset_name "$DATASET" \
      --data_dir "$DATA_DIR" \
      --result_dir "$RESULT_DIR" \
      --model_dir "$MODEL_DIR" \
      --train_size "$TRAIN_SIZE" \
      --val_size "$VAL_SIZE" \
      --test_size "$TEST_SIZE" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --ipm_type "$IPM_TYPE" \
      --beta "$BETA" \
      --z_dim "$DIM" \
      --h_dim1 "$DIM" \
      --h_dim2 "$DIM" \
      --num_layers1 "$LAYER" \
      --num_layers2 "$LAYER" \
      --exp_name "$EXP_NAME" \
      --verbose \
      2>&1 | tee "$LOG_FILE"
  done
done

echo "All runs completed."