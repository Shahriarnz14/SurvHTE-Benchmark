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
        for scenario in range(1, 2): # only one scenario per twin data
            try:
                result = pd.read_csv(path)
                if result is not None:
                    scenario_dict[f"scenario_{scenario}"] = result
            except Exception as e:
                # Log or ignore as needed
                continue
        experiment_setups[base_name] = scenario_dict


    output_pickle_path = f"results/real_data/models_causal_impute/meta_learner/{args.meta_learner}/"
    output_pickle_path += f"twin_{args.meta_learner}_{args.impute_method}_repeats_{args.num_repeats}.pkl"
    print("Output results path:", output_pickle_path)

    # base_regressors = ['ridge', 'lasso', 'rf', 'gbr', 'xgb']
    base_regressors = ['lasso', 'rf', 'xgb']
    if os.path.exists(output_pickle_path):
        print(f"Pickle file already exists. Loading from {output_pickle_path}...")
        with open(output_pickle_path, "rb") as f:
            results_dict = pickle.load(f)
    else:
        results_dict = {}

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

            for base_model in tqdm(base_regressors, desc="Base Models", leave=False):
                results_dict[setup_name][scenario_key][base_model] = {}
                start_time = time.time()

                for rand_idx in range(NUM_REPEATS_TO_INCLUDE):
                    X_train, W_train, Y_train, X_test, W_test, Y_test, cate_test_true = split_dict[rand_idx]
                    
                    max_time = Y_train[:, 0].max()
                    
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

                    # take first half of test set as validation set
                    X_val, W_val, Y_val = X_test[:int(len(dataset_df)*VAL_SIZE)], W_test[:int(len(dataset_df)*VAL_SIZE)], Y_test[:int(len(dataset_df)*VAL_SIZE)]
                    cate_val_true = cate_test_true[:int(len(dataset_df)*VAL_SIZE)]
                    X_test, W_test, Y_test = X_test[int(len(dataset_df)*VAL_SIZE):], W_test[int(len(dataset_df)*VAL_SIZE):], Y_test[int(len(dataset_df)*VAL_SIZE):]
                    cate_test_true = cate_test_true[int(len(dataset_df)*VAL_SIZE):]
                    Y_val_imputed = Y_test_imputed[:int(len(dataset_df)*VAL_SIZE)]
                    Y_test_imputed = Y_test_imputed[int(len(dataset_df)*VAL_SIZE):]
                    
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
                    mse_val, cate_val_pred, ate_val_pred = learner.evaluate(X_val, cate_val_true, W_val)


                    ate_true = TRUE_ATE.get((setup_name, scenario_key), cate_test_true.mean())
                    ate_true_val = TRUE_ATE.get((setup_name, scenario_key), cate_val_true.mean())

                    # Evaluate base regression models on test data
                    base_model_eval = learner.evaluate_test(X_test, Y_test_imputed, W_test)
                    base_model_eval_val = learner.evaluate_test(X_val, Y_val_imputed, W_val)

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

                        "cate_true_val": cate_val_true,
                        "cate_pred_val": cate_val_pred,
                        "ate_true_val": ate_true_val,
                        "ate_pred_val": ate_val_pred.mean_point,
                        "cate_mse_val": mse_val,
                        "ate_bias_val": ate_val_pred.mean_point - ate_true_val,
                        "base_model_eval_val": base_model_eval_val, # Store base model evaluation results
                        "ate_interval_val": ate_val_pred.conf_int_mean(),
                        "ate_statistics_val": ate_val_pred,
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
                results_dict[setup_name][scenario_key][base_model]["average"] = {
                    "mean_cate_mse": np.mean([avg[i]["cate_mse"] for i in range(NUM_REPEATS_TO_INCLUDE)]),
                    "std_cate_mse": np.std([avg[i]["cate_mse"] for i in range(NUM_REPEATS_TO_INCLUDE)]),
                    "mean_ate_pred": np.mean([avg[i]["ate_pred"] for i in range(NUM_REPEATS_TO_INCLUDE)]),
                    "std_ate_pred": np.std([avg[i]["ate_pred"] for i in range(NUM_REPEATS_TO_INCLUDE)]),
                    "mean_ate_true": np.mean([avg[i]["ate_true"] for i in range(NUM_REPEATS_TO_INCLUDE)]),
                    "std_ate_true": np.std([avg[i]["ate_true"] for i in range(NUM_REPEATS_TO_INCLUDE)]),
                    "mean_ate_bias": np.mean([avg[i]["ate_bias"] for i in range(NUM_REPEATS_TO_INCLUDE)]),
                    "std_ate_bias": np.std([avg[i]["ate_bias"] for i in range(NUM_REPEATS_TO_INCLUDE)]),
                    "base_model_eval": base_model_eval_performance,

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
                    
                    "runtime": (end_time - start_time) / len(range(NUM_REPEATS_TO_INCLUDE)),
                }

            with open(output_pickle_path, "wb") as f:
                pickle.dump(results_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--train_size", type=float, default='0.5')
    parser.add_argument("--impute_method", type=str, default="Pseudo_obs", choices=["Pseudo_obs", "Margin", "IPCW-T"])
    parser.add_argument("--meta_learner", type=str, default="t_learner", choices=["t_learner", "s_learner", "x_learner", "dr_learner"])
    parser.add_argument("--load_imputed", action="store_true")
    parser.add_argument("--imputed_path", type=str, default="real_data/imputed_times_lookup_twin.pkl")
    args = parser.parse_args()
    main(args)