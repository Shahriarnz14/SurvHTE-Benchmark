#!/bin/bash

# Usage:
# ./scripts/run_dml_learners_impute.sh double_ml 5000

# Get arguments or set defaults
# DML_LEARNER=${1:-causal_forest}
DML_LEARNER="double_ml"

# Constants
REPEATS=10
IMPUTED_PATH="real_data/imputed_times_lookup_actgL.pkl"
IMPUTE_METHODS=("Pseudo_obs" "Margin" "IPCW-T")

# Loop through each method and run
for METHOD in "${IMPUTE_METHODS[@]}"
do
    echo "Running with imputation method: $METHOD, dml-learner: $DML_LEARNER "
    python run_dml_learner_impute_actgL.py \
        --num_repeats $REPEATS \
        --train_size 0.75 \
        --dml_learner $DML_LEARNER \
        --impute_method $METHOD \
        --load_imputed \
        --imputed_path $IMPUTED_PATH
done