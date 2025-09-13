#!/bin/bash

# Set variables
REPEATS=10
IMPUTED_PATH="real_data/imputed_times_lookup_twin.pkl"
TRAIN_SIZE=0.5

# Define the imputation methods to loop over
IMPUTE_METHODS=("Pseudo_obs")
META_LEARNERs=("t_learner" "s_learner" "x_learner" "dr_learner")

# Loop through each method and run the Python script
for METHOD in "${IMPUTE_METHODS[@]}"
do
    for META_LEARNER in "${META_LEARNERs[@]}"
    do
        echo "Running $META_LEARNER with imputation method: $METHOD"
        python run_meta_learner_impute_twin.py \
            --num_repeats $REPEATS \
            --train_size $TRAIN_SIZE \
            --meta_learner $META_LEARNER \
            --impute_method $METHOD \
            --load_imputed \
            --imputed_path $IMPUTED_PATH
    done
done
