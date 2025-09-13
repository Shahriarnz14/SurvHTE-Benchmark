#!/bin/bash

# Usage:
# ./scripts/run_meta_learners_impute.sh t_learner 5000

# Get arguments or set defaults
META_LEARNER=${1:-t_learner}
TRAIN_SIZE=${2:-5000}

# Constants
REPEATS=10
TEST_SIZE=5000
IMPUTED_PATH="synthetic_data/imputed_times_lookup.pkl"
IMPUTE_METHODS=("Pseudo_obs" "Margin" "IPCW-T")

# Loop through each method and run
for METHOD in "${IMPUTE_METHODS[@]}"
do
    echo "Running with imputation method: $METHOD, meta-learner: $META_LEARNER, train size: $TRAIN_SIZE"
    python run_meta_learner_impute.py \
        --num_repeats $REPEATS \
        --train_size $TRAIN_SIZE \
        --test_size $TEST_SIZE \
        --meta_learner $META_LEARNER \
        --impute_method $METHOD \
        --load_imputed \
        --imputed_path $IMPUTED_PATH
done