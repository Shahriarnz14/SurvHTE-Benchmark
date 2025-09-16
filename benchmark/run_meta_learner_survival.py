import argparse
import os
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))
import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
from models_causal_survival_meta.meta_learners_survival import TLearnerSurvival, SLearnerSurvival, MatchingLearnerSurvival
from data import load_data, prepare_data_split
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from models_utils.checkpoint_utils import get_checkpoint_path


def main(args):
    # TODO: make the following args
    num_repeats = 10
    dataset_type = 'synthetic'
    cate_true_col = None
    train_size = 0.5
    val_size = 0.25
    test_size = 0.25
    data_dir = '/heinz-georgenas/users/xiaobins/SurvHTE-Benchmark/data' # TODO: !!!make sure such info is removed for Anonymity!!!

    experiment_setups, experiment_repeat_setups = load_data(dataset_type=dataset_type, data_dir=data_dir)

    output_pickle_path = f"results/semi_synthetic_data/models_causal_survival_meta/{args.meta_learner}/"
    output_pickle_path += f"actg_syn_{args.meta_learner}_{args.base_survival_model}_repeats_{num_repeats}.pkl"
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

    for config_name, setup_dict in tqdm(experiment_setups.items(), desc="Experiment Setups"):
        if config_name in results_dict:
            print(f"Skipping setup {config_name} as it already exists in results.")
            continue
        results_dict[config_name] = {}
        for scenario_key in tqdm(setup_dict, desc=f"{config_name} Scenarios"):
            dataset_df = setup_dict[scenario_key]["dataset"]
            dataset_summary = setup_dict[scenario_key]["summary"]
            split_dict = prepare_data_split(
                dataset_df, experiment_repeat_setups, 
                num_repeats=num_repeats, 
                dataset_type=dataset_type,
                cate_true_col=cate_true_col,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size
            )
            results_dict[config_name][scenario_key] = {}

            start_time = time.time()

            for rand_idx in range(num_repeats):
                X_train, W_train, Y_train, cate_true_train = split_dict[rand_idx]['train']
                X_val, W_val, Y_val, cate_true_val = split_dict[rand_idx]['val']
                X_test, W_test, Y_test, cate_true_test = split_dict[rand_idx]['test']
                
                max_time = Y_train[:, 0].max()
                ate_true = dataset_summary['ate']
                event_time_25pct = dataset_summary['event_time_25pct']
                event_time_50pct = dataset_summary['event_time_median']
                event_time_75pct = dataset_summary['event_time_75pct']
                
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

                # Generate checkpoint path
                checkpoint_path = get_checkpoint_path(
                    dataset_type='synthetic',
                    causal_config=config_name,
                    scenario=scenario_key,
                    model_family=args.meta_learner,
                    model_name=f"{args.meta_learner}_{base_model}_{args.survival_metric}",
                    repeat_idx=rand_idx
                )

                # Save the model
                learner.save_model(checkpoint_path)
                
                # Evaluate base survival models on test data
                base_model_eval = learner.evaluate_test(X_test, Y_test, W_test)
                base_model_eval_val = learner.evaluate_test(X_val, Y_val, W_val)
                
                # Evaluate causal effect predictions
                mse_test, cate_test_pred, ate_test_pred = learner.evaluate(X_test, cate_true_test, W_test)
                mse_val, cate_val_pred, ate_val_pred = learner.evaluate(X_val, cate_true_val, W_val)

                results_dict[config_name][scenario_key][rand_idx] = {
                    "cate_true": cate_true_test,
                    "cate_pred": cate_test_pred,
                    "ate_true": ate_true,
                    "ate_pred": ate_test_pred,
                    "cate_mse": mse_test,
                    "ate_bias": ate_test_pred - ate_true,
                    "base_model_eval": base_model_eval,  # Store base model evaluation results

                    # val set:
                    "cate_true_val": cate_true_val,
                    "cate_pred": cate_val_pred,
                    "ate_pred_val": ate_val_pred,
                    "cate_mse_val": mse_val,
                    "ate_bias_val": ate_val_pred - ate_true,
                    "base_model_eval_val": base_model_eval_val,  # Store base model evaluation results
                }

                # print(f"Completed {config_name}, {scenario_key}, repeat {rand_idx}: CATE MSE={mse_test:.4f}, ATE True={ate_true:.4f}, ATE Pred={ate_test_pred:.4f}")
                # loaded_learner = learner.load_model(checkpoint_path)
                # loaded_mse_test, cate_test_pred, loaded_ate_test_pred = loaded_learner.evaluate(X_test, cate_true_test, W_test)
                # print(f"Loaded model evaluation: CATE MSE={loaded_mse_test:.4f}, ATE True={ate_true:.4f}, ATE Pred={loaded_ate_test_pred:.4f}")

                # import pdb; pdb.set_trace()

            end_time = time.time()
            avg = results_dict[config_name][scenario_key]
            if len(avg) == 0:
                base_model_eval_performance = {}
            else:
                base_model_eval_performance = {
                                                base_model_k: 
                                                {
                                                    f"{stat}_{metric_j}": func([
                                                        avg[i]['base_model_eval'][base_model_k][metric_j] for i in range(num_repeats)
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
                                                        avg[i]['base_model_eval_val'][base_model_k][metric_j] for i in range(num_repeats)
                                                        if i in avg
                                                    ])
                                                    for metric_j in metric_j_dict
                                                    for stat, func in zip(['mean', 'std'], [np.nanmean, np.nanstd])
                                                }
                                                for base_model_k, metric_j_dict in avg[list(avg.keys())[0]]['base_model_eval_val'].items()
                                            }
                
            results_dict[config_name][scenario_key]["average"] = {
                "mean_cate_mse": np.mean([avg[i]["cate_mse"] for i in range(num_repeats) if i in avg]),
                "std_cate_mse": np.std([avg[i]["cate_mse"] for i in range(num_repeats) if i in avg]),
                "mean_ate_pred": np.mean([avg[i]["ate_pred"] for i in range(num_repeats) if i in avg]),
                "std_ate_pred": np.std([avg[i]["ate_pred"] for i in range(num_repeats) if i in avg]),
                "mean_ate_true": np.mean([avg[i]["ate_true"] for i in range(num_repeats) if i in avg]),
                "std_ate_true": np.std([avg[i]["ate_true"] for i in range(num_repeats) if i in avg]),
                "mean_ate_bias": np.mean([avg[i]["ate_bias"] for i in range(num_repeats) if i in avg]),
                "std_ate_bias": np.std([avg[i]["ate_bias"] for i in range(num_repeats) if i in avg]),
                "base_model_eval" : base_model_eval_performance,

                # val set:
                "mean_cate_mse_val": np.mean([avg[i]["cate_mse_val"] for i in range(num_repeats) if i in avg]),
                "std_cate_mse_val": np.std([avg[i]["cate_mse_val"] for i in range(num_repeats) if i in avg]),
                "mean_ate_pred_val": np.mean([avg[i]["ate_pred_val"] for i in range(num_repeats) if i in avg]),
                "std_ate_pred_val": np.std([avg[i]["ate_pred_val"] for i in range(num_repeats) if i in avg]),
                "mean_ate_true_val": np.mean([avg[i]["ate_true_val"] for i in range(num_repeats) if i in avg]),
                "std_ate_true_val": np.std([avg[i]["ate_true_val"] for i in range(num_repeats) if i in avg]),
                "mean_ate_bias_val": np.mean([avg[i]["ate_bias_val"] for i in range(num_repeats) if i in avg]),
                "std_ate_bias_val": np.std([avg[i]["ate_bias_val"] for i in range(num_repeats) if i in avg]),
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