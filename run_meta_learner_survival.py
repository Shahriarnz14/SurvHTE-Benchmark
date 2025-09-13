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

def load_scenario_data(h5_file_path, scenario_num):
    """Load data for a specific scenario from HDF5 file."""
    key = f"scenario_{scenario_num}/data"
    with pd.HDFStore(h5_file_path, mode='r') as store:
        if key not in store:
            return None
        df = store[key]
        metadata = store.get_storer(key).attrs.metadata
    return {"dataset": df, "metadata": metadata}

def prepare_data_split(dataset_df, experiment_repeat_setups, random_idx_col_list, num_training_data_points=5000, test_size=5000):
    """Prepare data splits for each random index column."""
    split_results = {}
    for rand_idx in random_idx_col_list:
        random_idx_vals = experiment_repeat_setups[rand_idx].values
        test_ids = random_idx_vals[-test_size:]
        train_ids = random_idx_vals[:min(num_training_data_points, len(random_idx_vals) - test_size)]

        X_cols = [c for c in dataset_df.columns if c.startswith("X") and c[1:].isdigit()]
        train_df = dataset_df[dataset_df['id'].isin(train_ids)]
        test_df = dataset_df[dataset_df['id'].isin(test_ids)]

        X_train = train_df[X_cols].to_numpy()
        W_train = train_df["W"].to_numpy()
        Y_train = train_df[["observed_time", "event"]].to_numpy()

        X_test = test_df[X_cols].to_numpy()
        W_test = test_df["W"].to_numpy()
        Y_test = test_df[["observed_time", "event"]].to_numpy()
        cate_test_true = (test_df["T1"] - test_df["T0"]).to_numpy()

        split_results[rand_idx] = (X_train, W_train, Y_train, X_test, W_test, Y_test, cate_test_true)
    return split_results

def main(args):
    """Main function to run the survival meta-learner experiments."""
    # Load experiment setups
    store_files = [
        "synthetic_data/RCT_0_5.h5",
        "synthetic_data/RCT_0_05.h5",
        "synthetic_data/e_X.h5",
        "synthetic_data/e_X_U.h5",
        "synthetic_data/e_X_no_overlap.h5",
        "synthetic_data/e_X_info_censor.h5",
        "synthetic_data/e_X_U_info_censor.h5",
        "synthetic_data/e_X_no_overlap_info_censor.h5"
    ]

    experiment_setups = {}
    for path in store_files:
        base_name = os.path.splitext(os.path.basename(path))[0]
        scenario_dict = {}
        # for scenario in range(1, 11):
        for scenario in ['A', 'B', 'C', 'D', 'E']:
            result = load_scenario_data(path, scenario)
            if result is not None:
                scenario_dict[f"scenario_{scenario}"] = result
        experiment_setups[base_name] = scenario_dict

    experiment_repeat_setups = pd.read_csv("synthetic_data/idx_split.csv").set_index("idx")
    random_idx_col_list = experiment_repeat_setups.columns.to_list()[:args.num_repeats]

    output_pickle_path = f"results/synthetic_data/models_causal_survival_meta/{args.meta_learner}/"
    output_pickle_path += f"{args.meta_learner}_{args.base_survival_model}_repeats_{args.num_repeats}_train_{args.train_size}.pkl"
    print("Output results path:", output_pickle_path)

    # Define base survival models to use
    base_model = args.base_survival_model
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
        results_dict[setup_name] = {}
        for scenario_key in tqdm(setup_dict, desc=f"{setup_name} Scenarios"):
            dataset_df = setup_dict[scenario_key]["dataset"]
            split_dict = prepare_data_split(dataset_df, experiment_repeat_setups, random_idx_col_list, args.train_size, args.test_size)
            results_dict[setup_name][scenario_key] = {}


            start_time = time.time()

            for rand_idx in random_idx_col_list:
                X_train, W_train, Y_train, X_test, W_test, Y_test, cate_test_true = split_dict[rand_idx]

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
                
                # Evaluate base survival models on test data
                base_model_eval = learner.evaluate_test(X_test, Y_test, W_test)
                
                # Evaluate causal effect predictions
                mse_test, cate_test_pred, ate_test_pred = learner.evaluate(X_test, cate_test_true, W_test)

                results_dict[setup_name][scenario_key][rand_idx] = {
                    "cate_true": cate_test_true,
                    "cate_pred": cate_test_pred,
                    "ate_true": cate_test_true.mean(),
                    "ate_pred": ate_test_pred,
                    "cate_mse": mse_test,
                    "ate_bias": ate_test_pred - cate_test_true.mean(),
                    "base_model_eval": base_model_eval  # Store base model evaluation results
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
                                                        avg[i]['base_model_eval'][base_model_k][metric_j] for i in random_idx_col_list
                                                        if i in avg
                                                    ])
                                                    for metric_j in metric_j_dict
                                                    for stat, func in zip(['mean', 'std'], [np.nanmean, np.nanstd])
                                                }
                                                for base_model_k, metric_j_dict in avg[list(avg.keys())[0]]['base_model_eval'].items()
                                            }

            results_dict[setup_name][scenario_key]["average"] = {
                "mean_cate_mse": np.mean([avg[i]["cate_mse"] for i in random_idx_col_list if i in avg]),
                "std_cate_mse": np.std([avg[i]["cate_mse"] for i in random_idx_col_list if i in avg]),
                "mean_ate_pred": np.mean([avg[i]["ate_pred"] for i in random_idx_col_list if i in avg]),
                "std_ate_pred": np.std([avg[i]["ate_pred"] for i in random_idx_col_list if i in avg]),
                "mean_ate_true": np.mean([avg[i]["ate_true"] for i in random_idx_col_list if i in avg]),
                "std_ate_true": np.std([avg[i]["ate_true"] for i in random_idx_col_list if i in avg]),
                "mean_ate_bias": np.mean([avg[i]["ate_bias"] for i in random_idx_col_list if i in avg]),
                "std_ate_bias": np.std([avg[i]["ate_bias"] for i in random_idx_col_list if i in avg]),
                "runtime": (end_time - start_time) / len(avg) if len(avg) > 0 else 0,
                "base_model_eval" : base_model_eval_performance
                }

            with open(output_pickle_path, "wb") as f:
                pickle.dump(results_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=5000)
    parser.add_argument("--survival_metric", type=str, default="mean", choices=["median", "mean"])
    parser.add_argument("--meta_learner", type=str, default="t_learner_survival", 
                        choices=["t_learner_survival", "s_learner_survival", "matching_learner_survival"])
    parser.add_argument("--base_survival_model", type=str, default="RandomSurvivalForest",
                        choices=["RandomSurvivalForest", "DeepSurv", "DeepHit"])
    parser.add_argument("--num_matches", type=int, default=5, help="Number of matches for matching learner")
    args = parser.parse_args()
    main(args)