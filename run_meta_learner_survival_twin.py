import argparse
import os
import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
from models_causal_survival_meta.meta_learners_survival import TLearnerSurvival, SLearnerSurvival, MatchingLearnerSurvival
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

NUM_REPEATS_TO_INCLUDE = 10
TRAIN_SIZE = 0.5
VAL_SIZE = 0.25
TEST_SIZE = 0.25

TRUE_ATE = {('twin', 'scenario_1'): 5.038157894736842,
            ('twin30', 'scenario_1'): 0.32824561403508773,
            ('twin180', 'scenario_1'): 2.260438596491228}

def prepare_twin_data_split(dataset_df, X_cols, W_col, cate_base_col, experiment_repeat_setup):
    split_results = {}
    length = len(experiment_repeat_setup)
    
    for rand_idx in range(NUM_REPEATS_TO_INCLUDE):
        y_cols = ['observed_time', 'event']
        # take the first half of the dataset for training and the second half for testing
        train_ids = experiment_repeat_setup[f'random_idx{rand_idx}'][:int(length*TRAIN_SIZE)].values
        test_ids =  experiment_repeat_setup[f'random_idx{rand_idx}'][int(length*TRAIN_SIZE):].values # this includes both validation and test data
        
        train_df = dataset_df[dataset_df['idx'].isin(train_ids)]
        test_df = dataset_df[dataset_df['idx'].isin(test_ids)]

        X_train = train_df[X_cols].to_numpy()
        W_train = train_df[W_col].to_numpy().flatten()
        Y_train = train_df[y_cols].to_numpy()

        X_test = test_df[X_cols].to_numpy()
        W_test = test_df[W_col].to_numpy().flatten()
        Y_test = test_df[y_cols].to_numpy()

        cate_test_true = test_df[cate_base_col].to_numpy()

        split_results[rand_idx] = (X_train, W_train, Y_train, X_test, W_test, Y_test, cate_test_true)

    return split_results

def main(args):
    """Main function to run the survival meta-learner experiments."""
    # Load experiment setups
    store_files = [
        "real_data/twin.csv",
        "real_data/twin30.csv",
        "real_data/twin180.csv",
    ]

    X_binary_cols = ['anemia', 'cardiac', 'lung', 'diabetes', 'herpes', 'hydra',
        'hemo', 'chyper', 'phyper', 'eclamp', 'incervix', 'pre4000', 'preterm',
        'renal', 'rh', 'uterine', 'othermr', 
        'gestat', 'dmage', 'dmeduc', 'dmar', 'nprevist', 'adequacy']
    X_num_cols = ['dtotord', 'cigar', 'drink', 'wtgain']
    X_ohe_cols = ['pldel_2', 'pldel_3', 'pldel_4', 'pldel_5', 'resstatb_2', 'resstatb_3', 'resstatb_4', 
                'mpcb_1', 'mpcb_2', 'mpcb_3', 'mpcb_4', 'mpcb_5', 'mpcb_6', 'mpcb_7', 'mpcb_8', 'mpcb_9']

    X_cols = X_binary_cols + X_num_cols + X_ohe_cols
    W_col = ['W']
    cate_true_col = 'true_cate'
    experiment_repeat_setups = [pd.read_csv(f'real_data/idx_split_twin.csv')]


    experiment_setups = {}
    for path in store_files:
        base_name = os.path.splitext(os.path.basename(path))[0]
        scenario_dict = {}
        for scenario in range(1, 2):
            result = pd.read_csv(path)
            if result is not None:
                scenario_dict[f"scenario_{scenario}"] = result
        experiment_setups[base_name] = scenario_dict

    output_pickle_path = f"results/real_data/models_causal_survival_meta/{args.meta_learner}/"
    output_pickle_path += f"twin_{args.meta_learner}_{args.base_survival_model}_repeats_{NUM_REPEATS_TO_INCLUDE}.pkl"
    print("Output results path:", output_pickle_path)

    # Define base survival models to use
    base_model = args.base_survival_model
    if os.path.exists(output_pickle_path):
        print(f"Pickle file already exists. Loading from {output_pickle_path}...")
        with open(output_pickle_path, "rb") as f:
            results_dict = pickle.load(f)
    else:
        results_dict = {}

    # Define hyperparameter grids for each model
    hyperparameter_grids = {
        'RandomSurvivalForest': {
            'n_estimators': [50, 100],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [3, 5]
        },
        'DeepSurv': {
            'num_nodes': [32, 64],
            'dropout': [0.1, 0.4],
            'lr': [0.01, 0.001],
            'epochs': [100, 500]
        },
        'DeepHit': {
            'num_nodes': [32, 64],
            'dropout': [0.1, 0.4],
            'lr': [0.01, 0.001],
            'epochs': [100, 500]
        }
    }

    for setup_name, setup_dict in tqdm(experiment_setups.items(), desc="Experiment Setups"):
        if setup_name in results_dict:
            print(f"Skipping setup {setup_name} as it already exists in results.")
            continue
        results_dict[setup_name] = {}
        experiment_repeat_setup = experiment_repeat_setups[0]
        for scenario_key in tqdm(setup_dict, desc=f"{setup_name} Scenarios"):
            dataset_df = setup_dict[scenario_key]
            split_dict = prepare_twin_data_split(dataset_df, X_cols, W_col, cate_true_col, 
                                             experiment_repeat_setup)
            results_dict[setup_name][scenario_key] = {}

            start_time = time.time()

            for rand_idx in range(NUM_REPEATS_TO_INCLUDE):
                X_train, W_train, Y_train, X_test, W_test, Y_test, cate_test_true = split_dict[rand_idx]
                # take first half of test set as validation set
                X_val, W_val, Y_val = X_test[:int(len(dataset_df)*VAL_SIZE)], W_test[:int(len(dataset_df)*VAL_SIZE)], Y_test[:int(len(dataset_df)*VAL_SIZE)]
                cate_val_true = cate_test_true[:int(len(dataset_df)*VAL_SIZE)]
                X_test, W_test, Y_test = X_test[int(len(dataset_df)*VAL_SIZE):], W_test[int(len(dataset_df)*VAL_SIZE):], Y_test[int(len(dataset_df)*VAL_SIZE):]
                cate_test_true = cate_test_true[int(len(dataset_df)*VAL_SIZE):]
                max_time = Y_train[:, 0].max()
                
                # Initialize the appropriate meta-learner
                if args.meta_learner == "t_learner_survival":
                    learner = TLearnerSurvival(
                        base_model_name=base_model,
                        base_model_grid=hyperparameter_grids,
                        metric=args.survival_metric,
                        max_time=max_time
                    )
                elif args.meta_learner == "s_learner_survival":
                    learner = SLearnerSurvival(
                        base_model_name=base_model,
                        base_model_grid=hyperparameter_grids,
                        metric=args.survival_metric,
                        max_time=max_time
                    )
                elif args.meta_learner == "matching_learner_survival":
                    learner = MatchingLearnerSurvival(
                        base_model_name=base_model,
                        base_model_grid=hyperparameter_grids,
                        metric=args.survival_metric,
                        num_matches=args.num_matches,
                        max_time=max_time
                    )

                if args.meta_learner == "t_learner_survival":
                    if Y_train[W_train == 1, 1].sum() <= 1:
                        print(f"[Warning]: For {args.meta_learner}, No event in treatment group. Skipping iteration {rand_idx}.")
                        continue
                    if Y_train[W_train == 0, 1].sum() <= 1:
                        print(f"[Warning]: For {args.meta_learner}, No event in control group. Skipping iteration {rand_idx}.")
                        continue

                # Fit the learner
                learner.fit(X_train, W_train, Y_train)

                ate_true = TRUE_ATE.get((setup_name, scenario_key), cate_test_true.mean())
                ate_true_val = TRUE_ATE.get((setup_name, scenario_key), cate_val_true.mean())
                
                # Evaluate base survival models on test data
                base_model_eval = learner.evaluate_test(X_test, Y_test, W_test)
                base_model_eval_val = learner.evaluate_test(X_val, Y_val, W_val)
                
                # Evaluate causal effect predictions
                mse_test, cate_test_pred, ate_test_pred = learner.evaluate(X_test, cate_test_true, W_test)
                mse_val, cate_val_pred, ate_val_pred = learner.evaluate(X_val, cate_val_true, W_val)

                results_dict[setup_name][scenario_key][rand_idx] = {
                    "cate_true": cate_test_true,
                    "cate_pred": cate_test_pred,
                    "ate_true": ate_true,
                    "ate_pred": ate_test_pred,
                    "cate_mse": mse_test,
                    "ate_bias": ate_test_pred - ate_true,
                    "base_model_eval": base_model_eval,  # Store base model evaluation results

                    # val set:
                    "cate_true_val": cate_val_true,
                    "cate_pred": cate_val_pred,
                    "ate_true_val": ate_true_val,
                    "ate_pred_val": ate_val_pred,
                    "cate_mse_val": mse_val,
                    "ate_bias_val": ate_val_pred - ate_true_val,
                    "base_model_eval_val": base_model_eval_val,  # Store base model evaluation results
                }

            end_time = time.time()
            avg = results_dict[setup_name][scenario_key]
            if len(avg) == 0:
                base_model_eval_performance = {}
            else:
                base_model_eval_performance = {
                                                base_model_k: 
                                                {
                                                    f"{stat}_{metric_j}": func([
                                                        avg[i]['base_model_eval'][base_model_k][metric_j] for i in range(NUM_REPEATS_TO_INCLUDE)
                                                        if i in avg
                                                    ])
                                                    for metric_j in metric_j_dict
                                                    for stat, func in zip(['mean', 'std'], [np.nanmean, np.nanstd])
                                                }
                                                for base_model_k, metric_j_dict in avg[list(avg.keys())[0]]['base_model_eval'].items()
                                            }
                base_model_eval_performance_val = {
                                                base_model_k: 
                                                {
                                                    f"{stat}_{metric_j}": func([
                                                        avg[i]['base_model_eval_val'][base_model_k][metric_j] for i in range(NUM_REPEATS_TO_INCLUDE)
                                                        if i in avg
                                                    ])
                                                    for metric_j in metric_j_dict
                                                    for stat, func in zip(['mean', 'std'], [np.nanmean, np.nanstd])
                                                }
                                                for base_model_k, metric_j_dict in avg[list(avg.keys())[0]]['base_model_eval_val'].items()
                                            }
                
            results_dict[setup_name][scenario_key]["average"] = {
                "mean_cate_mse": np.mean([avg[i]["cate_mse"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "std_cate_mse": np.std([avg[i]["cate_mse"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "mean_ate_pred": np.mean([avg[i]["ate_pred"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "std_ate_pred": np.std([avg[i]["ate_pred"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "mean_ate_true": np.mean([avg[i]["ate_true"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "std_ate_true": np.std([avg[i]["ate_true"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "mean_ate_bias": np.mean([avg[i]["ate_bias"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "std_ate_bias": np.std([avg[i]["ate_bias"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "base_model_eval" : base_model_eval_performance,

                # val set:
                "mean_cate_mse_val": np.mean([avg[i]["cate_mse_val"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "std_cate_mse_val": np.std([avg[i]["cate_mse_val"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "mean_ate_pred_val": np.mean([avg[i]["ate_pred_val"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "std_ate_pred_val": np.std([avg[i]["ate_pred_val"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "mean_ate_true_val": np.mean([avg[i]["ate_true_val"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "std_ate_true_val": np.std([avg[i]["ate_true_val"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "mean_ate_bias_val": np.mean([avg[i]["ate_bias_val"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "std_ate_bias_val": np.std([avg[i]["ate_bias_val"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "base_model_eval_val" : base_model_eval_performance_val,

                "runtime": (end_time - start_time) / len(avg) if len(avg) > 0 else 0,
                }

            with open(output_pickle_path, "wb") as f:
                pickle.dump(results_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--train_size", type=float, default=0.5)
    parser.add_argument("--survival_metric", type=str, default="mean", choices=["median", "mean"])
    parser.add_argument("--meta_learner", type=str, default="t_learner_survival", 
                        choices=["t_learner_survival", "s_learner_survival", "matching_learner_survival"])
    parser.add_argument("--base_survival_model", type=str, default="RandomSurvivalForest",
                        choices=["RandomSurvivalForest", "DeepSurv", "DeepHit"])
    parser.add_argument("--num_matches", type=int, default=5, help="Number of matches for matching learner")
    args = parser.parse_args()
    main(args)