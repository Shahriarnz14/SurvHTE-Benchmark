import argparse
import os
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))
import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
from models_causal_impute.dml_learners import CausalForest, DoubleML
from models_causal_impute.survival_eval_impute import SurvivalEvalImputer
from data import load_data, prepare_data_split
from models_utils.checkpoint_utils import get_checkpoint_path


def main(args):
    # TODO: make the following args
    num_repeats = 10
    dataset_type = 'synthetic'
    train_size = 0.5
    val_size = 0.25
    test_size = 0.25
    data_dir = '/heinz-georgenas/users/xiaobins/SurvHTE-Benchmark/data' # TODO: !!!make sure such info is removed for Anonymity!!!

    experiment_setups, experiment_repeat_setups = load_data(dataset_type=dataset_type, data_dir=data_dir)
    
    output_pickle_path = f"results/synthetic_data/models_causal_impute/dml_learner/{args.dml_learner}/"
    output_pickle_path += f"{args.dml_learner}_{args.impute_method}_repeats_{args.num_repeats}_train_{args.train_size}.pkl"
    print("Output results path:", output_pickle_path)

    # TODO: add: check if (1) already data imputed (2) model already trained
    results_dict = {}

    for config_name, setup_dict in tqdm(experiment_setups.items(), desc="Experiment Setups"):
        results_dict[config_name] = {}
        for scenario_key in tqdm(setup_dict, desc=f"{config_name} Scenarios"):
            dataset_df = setup_dict[scenario_key]["dataset"]
            dataset_summary = setup_dict[scenario_key]["summary"]
            split_dict = prepare_data_split(
                dataset_df, experiment_repeat_setups, 
                num_repeats=num_repeats, 
                dataset_type=dataset_type,
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

                if args.load_imputed:
                    with open(args.imputed_path, "rb") as f:
                        imputed_times = pickle.load(f)
                    imputed_results = imputed_times.get(args.impute_method, {}).get(config_name, {}).get(scenario_key, {}).get(args.train_size, {}).get(rand_idx, {})
                    Y_train_imputed = imputed_results.get("Y_train_imputed", None)
                    Y_test_imputed = imputed_results.get("Y_test_imputed", None)
                else:
                    Y_train_imputed = Y_test_imputed = None

                if Y_train_imputed is None:
                    survival_imputer = SurvivalEvalImputer(imputation_method=args.impute_method)
                    Y_train_imputed, Y_test_imputed = survival_imputer.fit_transform(Y_train, Y_test)

                if Y_test_imputed is None:
                    survival_imputer = SurvivalEvalImputer(imputation_method=args.impute_method)
                    _, Y_test_imputed = survival_imputer.fit_transform(Y_train, Y_test, impute_train=False)

                learner_cls = {"causal_forest": CausalForest, "double_ml": DoubleML}[args.dml_learner]
                learner = learner_cls()

                learner.fit(X_train, W_train, Y_train_imputed)
                mse_test, cate_test_pred, ate_test_pred = learner.evaluate(X_test, cate_true_test, W_test)

                # Generate checkpoint path
                checkpoint_path = get_checkpoint_path(
                    dataset_type=dataset_type,
                    causal_config=config_name,
                    scenario=scenario_key,
                    model_family=args.dml_learner,
                    model_name=f"{args.dml_learner}_{args.impute_method}",
                    repeat_idx=rand_idx
                )
                # Save the model
                learner.save_model(checkpoint_path)


                results_dict[config_name][scenario_key][rand_idx] = {
                    "cate_true": cate_true_test,
                    "cate_pred": cate_test_pred,
                    "ate_true": ate_true,
                    "ate_pred": ate_test_pred.mean_point,
                    "cate_mse": mse_test,
                    "ate_bias": ate_test_pred.mean_point - ate_true,
                    "ate_interval": ate_test_pred.conf_int_mean(),
                    "ate_statistics": ate_test_pred,
                }

                # print(f"Completed {config_name}, {scenario_key}, repeat {rand_idx}: CATE MSE={mse_test:.4f}, ATE True={ate_true:.4f}, ATE Pred={ate_test_pred.mean_point:.4f}")

                # # loading (must use the correct subclass)
                # loaded_learner = learner.load_model(checkpoint_path)
                # loaded_mse_test, cate_test_pred, loaded_ate_test_pred = loaded_learner.evaluate(X_test, cate_true_test, W_test)
                # print(f"Loaded model evaluation: CATE MSE={loaded_mse_test:.4f}, ATE True={ate_true:.4f}, ATE Pred={loaded_ate_test_pred.mean_point:.4f}")

                # import pdb; pdb.set_trace()

            end_time = time.time()

            if len(results_dict[config_name][scenario_key]) == 0:
                print(f"[Warning]: No valid results for {config_name}, {scenario_key}. Skipping.")
                continue

            # Save results to the setup dictionary
            avg = results_dict[config_name][scenario_key]
            results_dict[config_name][scenario_key]["average"] = {
                "mean_cate_mse": np.mean([avg[i]["cate_mse"] for i in range(num_repeats) if i in avg]),
                "std_cate_mse": np.std([avg[i]["cate_mse"] for i in range(num_repeats) if i in avg]),
                "mean_ate_pred": np.mean([avg[i]["ate_pred"] for i in range(num_repeats) if i in avg]),
                "std_ate_pred": np.std([avg[i]["ate_pred"] for i in range(num_repeats) if i in avg]),
                "mean_ate_true": np.mean([avg[i]["ate_true"] for i in range(num_repeats) if i in avg]),
                "std_ate_true": np.std([avg[i]["ate_true"] for i in range(num_repeats) if i in avg]),
                "mean_ate_bias": np.mean([avg[i]["ate_bias"] for i in range(num_repeats) if i in avg]),
                "std_ate_bias": np.std([avg[i]["ate_bias"] for i in range(num_repeats) if i in avg]),
                "runtime": (end_time - start_time) / len(avg) if len(avg) > 0 else 0,
            }

            with open(output_pickle_path, "wb") as f:
                pickle.dump(results_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=5000)
    parser.add_argument("--impute_method", type=str, default="Margin", choices=["Pseudo_obs", "Margin", "IPCW-T"])
    parser.add_argument("--dml_learner", type=str, default="causal_forest", choices=["double_ml", "causal_forest"])
    parser.add_argument("--load_imputed", action="store_true")
    parser.add_argument("--imputed_path", type=str, default="synthetic_data/imputed_times_lookup.pkl")
    args = parser.parse_args()
    main(args)