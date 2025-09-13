#!/bin/bash

# Usage:
# ./scripts/run_meta_learners_survival.sh t_learner_survival 5000

# Get arguments or set defaults
META_LEARNER=${1:-t_learner_survival}
TRAIN_SIZE=${2:-5000}

# Constants
REPEATS=10
TEST_SIZE=5000
NUM_MATCHES=5
SURVIVAL_METHODS=("RandomSurvivalForest" "DeepSurv" "DeepHit")

# Loop through each method and run
for SURVIVAL_METHOD in "${SURVIVAL_METHODS[@]}"
do
    echo "Running with survival method: $METHOD, meta-learner: $META_LEARNER, train size: $TRAIN_SIZE"
    python run_meta_learner_survival.py \
        --num_repeats $REPEATS \
        --train_size $TRAIN_SIZE \
        --test_size $TEST_SIZE \
        --meta_learner $META_LEARNER \
        --base_survival_model $SURVIVAL_METHOD \
        --num_matches $NUM_MATCHES
done