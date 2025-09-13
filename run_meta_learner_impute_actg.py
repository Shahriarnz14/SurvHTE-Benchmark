import argparse
import os
import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
from models_causal_impute.meta_learners import T_Learner, S_Learner, X_Learner, DR_Learner
from models_causal_impute.survival_eval_impute import SurvivalEvalImputer

NUM_REPEATS_TO_INCLUDE = 10

TRUE_ATE = {('ACTG_175_HIV1', 'scenario_1'): 2.7977461375268904,
            ('ACTG_175_HIV2', 'scenario_1'): 2.603510045518606,
            ('ACTG_175_HIV3', 'scenario_1'): 2.051686700212568}

def prepare_actg_data_split(dataset_df, X_cols, W_col, cate_base_col, experiment_repeat_setup):
    split_results = {}
    length = len(experiment_repeat_setup)
    
    for rand_idx in range(NUM_REPEATS_TO_INCLUDE):
        y_cols = [f't{rand_idx}', f'e{rand_idx}']
        # take the first half of the dataset for training and the second half for testing
        train_ids = experiment_repeat_setup[f'random_idx{rand_idx}'][:int(length*args.train_size)].values
        test_ids =  experiment_repeat_setup[f'random_idx{rand_idx}'][int(length*args.train_size):].values
        # test_ids = dataset_df['id'] # same as train_ids
        # train_ids = dataset_df['id']
        
        train_df = dataset_df[dataset_df['id'].isin(train_ids)]
        test_df = dataset_df[dataset_df['id'].isin(test_ids)]

        X_train = train_df[X_cols].to_numpy()
        W_train = train_df[W_col].to_numpy()
        Y_train = train_df[y_cols].to_numpy()

        X_test = test_df[X_cols].to_numpy()
        W_test = test_df[W_col].to_numpy()
        Y_test = test_df[y_cols].to_numpy()

        cate_test_true = test_df[cate_base_col].to_numpy()

        split_results[rand_idx] = (X_train, W_train, Y_train, X_test, W_test, Y_test, cate_test_true)

    return split_results

def summarize_experiment_results(results_dict):
    records = []
    for setup_name, setup_dict in results_dict.items():
        for scenario_key in setup_dict:
            row = {
                ("setup_name", ""): setup_name,
                ("scenario_key", ""): scenario_key
            }
            for base_model in setup_dict[scenario_key]:
                avg_result = setup_dict[scenario_key].get(base_model, {}).get("average", {})
                mean_mse = avg_result.get("mean_cate_mse", np.nan)
                std_mse = avg_result.get("std_cate_mse", np.nan)
                mean_ate_pred = avg_result.get("mean_ate_pred", np.nan)
                std_ate_pred = avg_result.get("std_ate_pred", np.nan)
                mean_ate_true = avg_result.get("mean_ate_true", np.nan)
                std_ate_true = avg_result.get("std_ate_true", np.nan)
                mean_ate_bias = avg_result.get("mean_ate_bias", np.nan)
                std_ate_bias = avg_result.get("std_ate_bias", np.nan)
                runtime = avg_result.get("runtime", np.nan)

                row[(base_model, "CATE_MSE")] = f"{mean_mse:.3f} ± {std_mse:.3f}" if not pd.isna(mean_mse) else np.nan
                row[(base_model, "ATE_pred")] = f"{mean_ate_pred:.3f} ± {std_ate_pred:.3f}" if not pd.isna(mean_ate_pred) else np.nan
                row[(base_model, "ATE_true")] = f"{mean_ate_true:.3f} ± {std_ate_true:.3f}" if not pd.isna(mean_ate_true) else np.nan
                row[(base_model, "ATE_bias")] = f"{mean_ate_bias:.3f} ± {std_ate_bias:.3f}" if not pd.isna(mean_ate_bias) else np.nan
                row[(base_model, "runtime [s]")] = round(runtime) if not pd.isna(runtime) else np.nan

            records.append(row)
    df = pd.DataFrame.from_records(records)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df

def main(args):
    # Load experiment setups
    store_files = [
        "real_data/ACTG_175_HIV1.csv",
        "real_data/ACTG_175_HIV2.csv",
        "real_data/ACTG_175_HIV3.csv",
    ]

    X_bi_cols = ['gender', 'race', 'hemo', 'homo', 'drugs', 'str2', 'symptom']
    X_cont_cols = ['age', 'wtkg',  'karnof', 'cd40', 'cd80']
    U = ['z30']
    W = ['trt']
    y_cols = ['observed_time_month', 'effect_non_censor'] # ['time', 'cid']

    X_cols = X_bi_cols + X_cont_cols
    W_col = W[0]
    cate_base_col = 'cate_base'
    experiment_repeat_setups = [pd.read_csv(f'real_data/idx_split_HIV{i}.csv') for i in range(1, 4)]


    experiment_setups = {}
    for path in store_files:
        base_name = os.path.splitext(os.path.basename(path))[0]
        scenario_dict = {}
        for scenario in range(1, 2): # only one scenario per HIV data
            try:
                result = pd.read_csv(path)
                if result is not None:
                    scenario_dict[f"scenario_{scenario}"] = result
            except Exception as e:
                # Log or ignore as needed
                continue
        experiment_setups[base_name] = scenario_dict


    output_pickle_path = f"results/real_data/models_causal_impute/meta_learner/{args.meta_learner}/"
    output_pickle_path += f"actg_{args.meta_learner}_{args.impute_method}_repeats_{args.num_repeats}.pkl"
    print("Output results path:", output_pickle_path)

    # base_regressors = ['ridge', 'lasso', 'rf', 'gbr', 'xgb']
    base_regressors = ['lasso', 'rf', 'xgb']
    results_dict = {}

    for setup_name, setup_dict in tqdm(experiment_setups.items(), desc="Experiment Setups"):
        results_dict[setup_name] = {}
        hiv_dataset_idx = int(setup_name[-1])
        experiment_repeat_setup = experiment_repeat_setups[hiv_dataset_idx-1]
        for scenario_key in tqdm(setup_dict, desc=f"{setup_name} Scenarios"):
            dataset_df = setup_dict[scenario_key]
            split_dict = prepare_actg_data_split(dataset_df, X_cols, W_col, cate_base_col, experiment_repeat_setup)
            results_dict[setup_name][scenario_key] = {}

            for base_model in tqdm(base_regressors, desc="Base Models", leave=False):
                results_dict[setup_name][scenario_key][base_model] = {}
                start_time = time.time()

                for rand_idx in range(NUM_REPEATS_TO_INCLUDE):
                    X_train, W_train, Y_train, X_test, W_test, Y_test, cate_test_true = split_dict[rand_idx]

                    if args.load_imputed:
                        with open(args.imputed_path, "rb") as f:
                            imputed_times = pickle.load(f)
                        imputed_results = imputed_times.get(args.impute_method, {}).get(setup_name, {}).get(scenario_key, {}).get(f'{str(int(args.train_size*100))}%', {}).get(rand_idx, {})
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
                    
                    if args.meta_learner in ["t_learner", "x_learner"]:
                        if Y_train[W_train == 1, 1].sum() <= 1:
                            print(f"[Warning]: For {args.meta_learner}, No event in treatment group. Skipping iteration {rand_idx}.")
                            continue
                        if Y_train[W_train == 0, 1].sum() <= 1:
                            print(f"[Warning]: For {args.meta_learner}, No event in control group. Skipping iteration {rand_idx}.")
                            continue

                    learner_cls = {"t_learner": T_Learner, "s_learner": S_Learner, "x_learner": X_Learner, "dr_learner": DR_Learner}[args.meta_learner]
                    learner = learner_cls(base_model_name=base_model)

                    learner.fit(X_train, W_train, Y_train_imputed)
                    mse_test, cate_test_pred, ate_test_pred = learner.evaluate(X_test, cate_test_true, W_test)

                    ate_true = TRUE_ATE.get((setup_name, scenario_key))

                    # Evaluate base regression models on test data
                    base_model_eval = learner.evaluate_test(X_test, Y_test_imputed, W_test)

                    results_dict[setup_name][scenario_key][base_model][rand_idx] = {
                        "cate_true": cate_test_true,
                        "cate_pred": cate_test_pred,
                        "ate_true": ate_true,
                        "ate_pred": ate_test_pred.mean_point,
                        "cate_mse": mse_test,
                        "ate_bias": ate_test_pred.mean_point - ate_true,
                        "base_model_eval": base_model_eval, # Store base model evaluation results
                        "ate_interval": ate_test_pred.conf_int_mean(),
                        "ate_statistics": ate_test_pred,
                    }

                end_time = time.time()
                
                if len(results_dict[setup_name][scenario_key][base_model]) == 0:
                    print(f"[Warning]: No valid results for {setup_name}, {scenario_key}, {base_model}. Skipping.")
                    continue

                avg = results_dict[setup_name][scenario_key][base_model]
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
                results_dict[setup_name][scenario_key][base_model]["average"] = {
                    "mean_cate_mse": np.mean([avg[i]["cate_mse"] for i in range(NUM_REPEATS_TO_INCLUDE)]),
                    "std_cate_mse": np.std([avg[i]["cate_mse"] for i in range(NUM_REPEATS_TO_INCLUDE)]),
                    "mean_ate_pred": np.mean([avg[i]["ate_pred"] for i in range(NUM_REPEATS_TO_INCLUDE)]),
                    "std_ate_pred": np.std([avg[i]["ate_pred"] for i in range(NUM_REPEATS_TO_INCLUDE)]),
                    "mean_ate_true": np.mean([avg[i]["ate_true"] for i in range(NUM_REPEATS_TO_INCLUDE)]),
                    "std_ate_true": np.std([avg[i]["ate_true"] for i in range(NUM_REPEATS_TO_INCLUDE)]),
                    "mean_ate_bias": np.mean([avg[i]["ate_bias"] for i in range(NUM_REPEATS_TO_INCLUDE)]),
                    "std_ate_bias": np.std([avg[i]["ate_bias"] for i in range(NUM_REPEATS_TO_INCLUDE)]),
                    "runtime": (end_time - start_time) / len(range(NUM_REPEATS_TO_INCLUDE)),
                    "base_model_eval": base_model_eval_performance
                }

            with open(output_pickle_path, "wb") as f:
                pickle.dump(results_dict, f)
            # break
        # break

    # df = summarize_experiment_results(results_dict)
    # print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--train_size", type=float, default='0.75')
    parser.add_argument("--impute_method", type=str, default="Pseudo_obs", choices=["Pseudo_obs", "Margin", "IPCW-T"])
    parser.add_argument("--meta_learner", type=str, default="t_learner", choices=["t_learner", "s_learner", "x_learner", "dr_learner"])
    parser.add_argument("--load_imputed", action="store_true")
    parser.add_argument("--imputed_path", type=str, default="real_data/imputed_times_lookup.pkl")
    args = parser.parse_args()
    main(args)