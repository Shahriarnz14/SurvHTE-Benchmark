#!/bin/bash

# Usage:
# ./scripts/run_meta_learners_impute.sh t_learner 5000 Pseudo_obs

# Get arguments or set defaults
META_LEARNER=${1:-t_learner}
TRAIN_SIZE=${2:-5000}
IMPUTE_METHOD=${3:-Pseudo_obs}

# Constants
REPEATS=10
TEST_SIZE=5000
IMPUTED_PATH="synthetic_data/imputed_times_lookup.pkl"

echo "Running with imputation method: $IMPUTE_METHOD, meta-learner: $META_LEARNER, train size: $TRAIN_SIZE"
python run_meta_learner_impute.py \
    --num_repeats $REPEATS \
    --train_size $TRAIN_SIZE \
    --test_size $TEST_SIZE \
    --meta_learner $META_LEARNER \
    --impute_method $IMPUTE_METHOD \
    --load_imputed \
    --imputed_path $IMPUTED_PATH
