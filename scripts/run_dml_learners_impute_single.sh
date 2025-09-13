#!/bin/bash

# Usage:
# ./scripts/run_dml_learners_impute.sh double_ml 5000 Pseudo_obs

# Get arguments or set defaults
DML_LEARNER=${1:-causal_forest}
TRAIN_SIZE=${2:-5000}
IMPUTE_METHOD=${3:-Pseudo_obs}

# Constants
REPEATS=10
TEST_SIZE=5000
IMPUTED_PATH="synthetic_data/imputed_times_lookup.pkl"

echo "Running with imputation method: $IMPUTE_METHOD, dml-learner: $DML_LEARNER, train size: $TRAIN_SIZE"
python run_dml_learner_impute.py \
    --num_repeats $REPEATS \
    --train_size $TRAIN_SIZE \
    --test_size $TEST_SIZE \
    --dml_learner $DML_LEARNER \
    --impute_method $IMPUTE_METHOD \
    --load_imputed \
    --imputed_path $IMPUTED_PATH
