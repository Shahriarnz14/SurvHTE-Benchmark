#!/bin/bash

# Usage:
# ./scripts/run_dml_learners_impute.sh

DATA_DIR="./data"
RESULT_DIR="./results"
DATASET="synthetic"
NUM_REPEATS=10
TRAIN_SIZE=5000
VAL_SIZE=2500
TEST_SIZE=2500

for impute in "Pseudo_obs" "Margin" "IPCW-T"; do
  for learner in "double_ml" "causal_forest"; do
    echo "Running with impute_method=${impute}, dml_learner=${learner}"
    python benchmark/run_dml_learner_impute.py \
      --num_repeats ${NUM_REPEATS} \
      --dataset_name ${DATASET} \
      --data_dir ${DATA_DIR} \
      --result_dir ${RESULT_DIR} \
      --train_size ${TRAIN_SIZE} \
      --val_size ${VAL_SIZE} \
      --test_size ${TEST_SIZE} \
      --impute_method ${impute} \
      --dml_learner ${learner}
  done
done