#!/bin/bash

# Usage:
# ./run_meta_learners_survival.sh t_learner 5000

# Get arguments or set defaults
SURVIVAL_METHODS=("RandomSurvivalForest" "DeepSurv" "DeepHit")
META_LEARNERS=("t_learner_survival" "s_learner_survival" "matching_learner_survival")

# Constants
REPEATS=10
NUM_MATCHES=5
TRAIN_SIZE=0.75

# Loop through each method and run
for SURVIVAL_METHOD in "${SURVIVAL_METHODS[@]}"
do
    for META_LEARNER in "${META_LEARNERS[@]}"
    do
        echo "Running with survival method: $SURVIVAL_METHOD, meta-learner: $META_LEARNER"
        python run_meta_learner_survival_actgL.py \
            --num_repeats $REPEATS \
            --train_size $TRAIN_SIZE \
            --meta_learner $META_LEARNER \
            --base_survival_model $SURVIVAL_METHOD \
            --num_matches $NUM_MATCHES
    done
done