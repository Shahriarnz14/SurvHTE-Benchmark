#!/bin/bash

# Usage:
# ./scripts/run_meta_learners_impute.sh

DATA_DIR="./data"
RESULT_DIR="./results"
DATASET="mimic_syn"
NUM_REPEATS=10
TRAIN_SIZE=0.5
VAL_SIZE=0.25
TEST_SIZE=0.25

for impute in "Pseudo_obs" "Margin" "IPCW-T"; do
  for learner in "t_learner" "s_learner" "x_learner" "dr_learner"; do
    echo "=============================================="
    echo "Running with impute_method=${impute}, meta_learner=${learner}"
    echo "=============================================="

    python benchmark/run_meta_learner_impute.py \
      --num_repeats ${NUM_REPEATS} \
      --dataset_name ${DATASET} \
      --data_dir ${DATA_DIR} \
      --result_dir ${RESULT_DIR} \
      --train_size ${TRAIN_SIZE} \
      --val_size ${VAL_SIZE} \
      --test_size ${TEST_SIZE} \
      --impute_method ${impute} \
      --meta_learner ${learner}
  done
done