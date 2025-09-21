#!/bin/bash
# run_csf_experiments.sh
# Script to run causal survival forest experiments one by one

# Fail immediately if a command fails
set -e

# Define workspace root (adjust if needed)
DATA_DIR="./data"
RESULT_DIR="./results"
NUM_REPEATS=10

echo "Running csf-synthetic..."
python benchmark/run_causal_survival_forest.py \
    --num_repeats ${NUM_REPEATS} \
    --dataset_name synthetic \
    --data_dir ${DATA_DIR} \
    --result_dir ${RESULT_DIR} \
    --train_size 5000 \
    --val_size 2500 \
    --test_size 2500

echo "Running csf-actg_syn..."
python benchmark/run_causal_survival_forest.py \
    --num_repeats ${NUM_REPEATS} \
    --dataset_name actg_syn \
    --data_dir ${DATA_DIR} \
    --result_dir ${RESULT_DIR} \
    --train_size 0.5 \
    --val_size 0.25 \
    --test_size 0.25

echo "Running csf-mimic_syn..."
python benchmark/run_causal_survival_forest.py \
    --num_repeats ${NUM_REPEATS} \
    --dataset_name mimic_syn \
    --data_dir ${DATA_DIR} \
    --result_dir ${RESULT_DIR} \
    --train_size 0.5 \
    --val_size 0.25 \
    --test_size 0.25

echo "Running csf-twin..."
python benchmark/run_causal_survival_forest.py \
    --num_repeats ${NUM_REPEATS} \
    --dataset_name twin \
    --data_dir ${DATA_DIR} \
    --result_dir ${RESULT_DIR} \
    --train_size 0.5 \
    --val_size 0.25 \
    --test_size 0.25