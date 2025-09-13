import argparse
import os
import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
from models_causal_impute.dml_learners import CausalForest, DoubleML
from models_causal_impute.survival_eval_impute import SurvivalEvalImputer


def load_scenario_data(h5_file_path, scenario_num):
    key = f"scenario_{scenario_num}/data"
    with pd.HDFStore(h5_file_path, mode='r') as store:
        if key not in store:
            return None
        df = store[key]
        metadata = store.get_storer(key).attrs.metadata
    return {"dataset": df, "metadata": metadata}


TRUE_ATE = {('RCT_0_5', 'scenario_B'): 0.124969, ('RCT_0_5', 'scenario_A'): 0.163441, ('RCT_0_5', 'scenario_C'): 0.74996,
            ('RCT_0_5', 'scenario_E'): 0.7537, ('RCT_0_5', 'scenario_D'): 0.723925,
            ('RCT_0_05', 'scenario_B'): 0.124969, ('RCT_0_05', 'scenario_A'): 0.163441, ('RCT_0_05', 'scenario_C'): 0.74996,
            ('RCT_0_05', 'scenario_E'): 0.7537, ('RCT_0_05', 'scenario_D'): 0.723925,
            ('e_X', 'scenario_B'): 0.124969, ('e_X', 'scenario_A'): 0.163441, ('e_X', 'scenario_C'): 0.74996,
            ('e_X', 'scenario_E'): 0.7537, ('e_X', 'scenario_D'): 0.723925,
            ('e_X_U', 'scenario_B'): 0.131728, ('e_X_U', 'scenario_A'): 0.003744, ('e_X_U', 'scenario_C'): 0.74036,
            ('e_X_U', 'scenario_E'): 0.74032, ('e_X_U', 'scenario_D'): 0.830668,
            ('e_X_no_overlap', 'scenario_B'): 0.124969, ('e_X_no_overlap', 'scenario_A'): 0.163441, ('e_X_no_overlap', 'scenario_C'): 0.74996,
            ('e_X_no_overlap', 'scenario_E'): 0.7537, ('e_X_no_overlap', 'scenario_D'): 0.723925,
            ('e_X_info_censor', 'scenario_B'): 0.124969, ('e_X_info_censor', 'scenario_A'): 0.163441, ('e_X_info_censor', 'scenario_C'): 0.74996,
            ('e_X_info_censor', 'scenario_E'): 0.7537, ('e_X_info_censor', 'scenario_D'): 0.723925,
            ('e_X_U_info_censor', 'scenario_B'): 0.131728, ('e_X_U_info_censor', 'scenario_A'): 0.003744, ('e_X_U_info_censor', 'scenario_C'): 0.74036,
            ('e_X_U_info_censor', 'scenario_E'): 0.74032, ('e_X_U_info_censor', 'scenario_D'): 0.830668,
            ('e_X_no_overlap_info_censor', 'scenario_B'): 0.124969, ('e_X_no_overlap_info_censor', 'scenario_A'): 0.163441, 
            ('e_X_no_overlap_info_censor', 'scenario_C'): 0.74996, ('e_X_no_overlap_info_censor', 'scenario_E'): 0.7537, 
            ('e_X_no_overlap_info_censor', 'scenario_D'): 0.723925}


def prepare_data_split(dataset_df, experiment_repeat_setups, random_idx_col_list, num_training_data_points=5000, test_size=5000):
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
    
    output_pickle_path = f"results/synthetic_data/models_causal_impute/dml_learner/{args.dml_learner}/"
    output_pickle_path += f"{args.dml_learner}_{args.impute_method}_repeats_{args.num_repeats}_train_{args.train_size}.pkl"
    print("Output results path:", output_pickle_path)

    results_dict = {}

    for setup_name, setup_dict in tqdm(experiment_setups.items(), desc="Experiment Setups"):
        results_dict[setup_name] = {}
        for scenario_key in tqdm(setup_dict, desc=f"{setup_name} Scenarios"):
            dataset_df = setup_dict[scenario_key]["dataset"]
            split_dict = prepare_data_split(dataset_df, experiment_repeat_setups, random_idx_col_list, args.train_size, args.test_size)
            results_dict[setup_name][scenario_key] = {}

            start_time = time.time()

            for rand_idx in random_idx_col_list:
                X_train, W_train, Y_train, X_test, W_test, Y_test, cate_test_true = split_dict[rand_idx]

                if args.load_imputed:
                    with open(args.imputed_path, "rb") as f:
                        imputed_times = pickle.load(f)
                    imputed_results = imputed_times.get(args.impute_method, {}).get(setup_name, {}).get(scenario_key, {}).get(args.train_size, {}).get(rand_idx, {})
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
                mse_test, cate_test_pred, ate_test_pred = learner.evaluate(X_test, cate_test_true, W_test)

                ate_true = TRUE_ATE.get((setup_name, scenario_key), cate_test_true.mean())

                results_dict[setup_name][scenario_key][rand_idx] = {
                    "cate_true": cate_test_true,
                    "cate_pred": cate_test_pred,
                    "ate_true": ate_true,
                    "ate_pred": ate_test_pred.mean_point,
                    "cate_mse": mse_test,
                    "ate_bias": ate_test_pred.mean_point - ate_true,
                    "ate_interval": ate_test_pred.conf_int_mean(),
                    "ate_statistics": ate_test_pred,
                }

            end_time = time.time()

            if len(results_dict[setup_name][scenario_key]) == 0:
                print(f"[Warning]: No valid results for {setup_name}, {scenario_key}. Skipping.")
                continue

            # Save results to the setup dictionary
            avg = results_dict[setup_name][scenario_key]
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
            }

            with open(output_pickle_path, "wb") as f:
                pickle.dump(results_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=5000)
    parser.add_argument("--impute_method", type=str, default="Pseudo_obs", choices=["Pseudo_obs", "Margin", "IPCW-T"])
    parser.add_argument("--dml_learner", type=str, default="causal_forest", choices=["double_ml", "causal_forest"])
    parser.add_argument("--load_imputed", action="store_true")
    parser.add_argument("--imputed_path", type=str, default="synthetic_data/imputed_times_lookup.pkl")
    args = parser.parse_args()
    main(args)