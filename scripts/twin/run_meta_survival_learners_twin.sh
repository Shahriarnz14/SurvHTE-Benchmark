#!/bin/bash

DATA_DIR="./data"
RESULT_DIR="./results"
DATASET="twin"
NUM_REPEATS=10
TRAIN_SIZE=0.5
VAL_SIZE=0.25
TEST_SIZE=0.25

set -e  # exit immediately if a command fails

META_LEARNERS=("t_learner_survival" "s_learner_survival" "matching_learner_survival")
BASE_MODELS=("RandomSurvivalForest" "DeepSurv" "DeepHit")

for meta in "${META_LEARNERS[@]}"; do
  for model in "${BASE_MODELS[@]}"; do
    echo "============================================================"
    echo "Running meta_learner: $meta | base_survival_model: $model"
    echo "============================================================"

    CMD=(
      python benchmark/run_meta_learner_survival.py
        -num_repeats ${NUM_REPEATS} \
        --dataset_name ${DATASET} \
        --data_dir ${DATA_DIR} \
        --result_dir ${RESULT_DIR} \
        --train_size ${TRAIN_SIZE} \
        --val_size ${VAL_SIZE} \
        --test_size ${TEST_SIZE} \
        --survival_metric mean
        --meta_learner "$meta"
        --base_survival_model "$model"
        --num_matches 5
    )

    # Add --save_model only for DeepSurv and DeepHit
    if [[ "$model" == "DeepSurv" || "$model" == "DeepHit" ]]; then
      CMD+=("--save_model")
    fi

    # Run the command
    "${CMD[@]}"

  done
done

echo "All meta-learner survival experiments finished!"