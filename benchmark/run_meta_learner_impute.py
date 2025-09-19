import argparse
import os
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))
import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
from models_causal_impute.meta_learners import T_Learner, S_Learner, X_Learner, DR_Learner
from models_causal_impute.survival_eval_impute import SurvivalEvalImputer
from data import load_data, prepare_data_split
from models_utils.checkpoint_utils import get_checkpoint_path


def main(args):
    num_repeats = args.num_repeats
    dataset_name = args.dataset_name
    dataset_type = (
        "synthetic" if dataset_name == "synthetic"
        else "semi-synthetic" if dataset_name in ["mimic_syn", "actg_syn"]
        else "real"
    )
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    data_dir = args.data_dir
    result_dir = args.result_dir
    train_size_str = f'{int(train_size*100)}%' if train_size<1 else f'{int(train_size)}'

    experiment_setups, experiment_repeat_setups = load_data(dataset_name=dataset_name, data_dir=data_dir)

    imputed_path = os.path.join(data_dir, dataset_type, f'imputed_times_lookup_{dataset_name}.pkl')
    output_pickle_path = os.path.join(result_dir, dataset_type, f'models_causal_impute/meta_learner/{args.meta_learner}/')
    output_pickle_path += f"{dataset_name}_{args.meta_learner}_{args.impute_method}_repeats_{args.num_repeats}_train_{train_size_str}.pkl"
    os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
    print("Output results path:", output_pickle_path)

    # base_regressors = ['ridge', 'lasso', 'rf', 'gbr', 'xgb']
    base_regressors = ['lasso', 'rf', 'xgb']

    if os.path.exists(output_pickle_path):
        print("Loading results from existing file.")
        with open(output_pickle_path, 'rb') as f:
            results_dict = pickle.load(f)
    else:
        print("Results file not found, creating new file.")
        results_dict = {}

    for config_name, setup_dict in tqdm(experiment_setups.items(), desc="Experiment Setups"):
        if config_name not in results_dict:
            results_dict[config_name] = {}
        for scenario_key in tqdm(setup_dict, desc=f"{config_name} Scenarios"):
            dataset_df = setup_dict[scenario_key]["dataset"]
            dataset_summary = setup_dict[scenario_key]["summary"]
            split_dict = prepare_data_split(
                dataset_df, experiment_repeat_setups, 
                num_repeats=num_repeats, 
                dataset_name=dataset_name,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size
            )
            if scenario_key not in results_dict[config_name]:
                results_dict[config_name][scenario_key] = {}

            for base_model in tqdm(base_regressors, desc="Base Models", leave=False):
                if base_model not in results_dict[config_name][scenario_key]:
                    results_dict[config_name][scenario_key][base_model] = {}

                for rand_idx in range(num_repeats):
                    X_train, W_train, Y_train, cate_true_train = split_dict[rand_idx]['train']
                    X_val, W_val, Y_val, cate_true_val = split_dict[rand_idx]['val']
                    X_test, W_test, Y_test, cate_true_test = split_dict[rand_idx]['test']
                    val_size_ = Y_val.shape[0]
                    Y_val_test = np.vstack((Y_val, Y_test))
                    
                    max_time = Y_train[:, 0].max()
                    ate_true = dataset_summary['ate']
                    event_time_25pct = dataset_summary['event_time_25pct']
                    event_time_50pct = dataset_summary['event_time_median']
                    event_time_75pct = dataset_summary['event_time_75pct']

                    # Generate checkpoint path
                    checkpoint_path = get_checkpoint_path(
                        dataset_type=dataset_type,
                        causal_config=config_name,
                        scenario=scenario_key,
                        model_family=args.meta_learner,
                        model_name=f"{args.meta_learner}_{base_model}_{args.impute_method}",
                        repeat_idx=rand_idx
                    )

                    learner_cls = {"t_learner": T_Learner, "s_learner": S_Learner, 
                                   "x_learner": X_Learner, "dr_learner": DR_Learner}[args.meta_learner]
                    learner = learner_cls(base_model_name=base_model)

                    if os.path.exists(checkpoint_path) and rand_idx in results_dict[config_name][scenario_key][base_model]:
                        print(f'Loading from model checkpoint {checkpoint_path}', end=' ')
                        learner.load_model(checkpoint_path)
                        runtime = results_dict[config_name][scenario_key][base_model][rand_idx]["runtime"]
                        with open(imputed_path, "rb") as f:
                            imputed_times = pickle.load(f)
                        imputed_results = \
                            imputed_times.get(args.impute_method, {})\
                                .get(config_name, {}).get(scenario_key, {})\
                                    .get(train_size_str, {}).get(rand_idx, {})
                        Y_train_imputed = imputed_results.get("Y_train_imputed", None)
                        Y_val_imputed = imputed_results.get("Y_val_imputed", None)
                        Y_test_imputed = imputed_results.get("Y_test_imputed", None)
                        print(f'run time {runtime:.0f} seconds')

                    else:
                        start_time = time.time()

                        if args.load_imputed:
                            with open(imputed_path, "rb") as f:
                                imputed_times = pickle.load(f)
                            imputed_results = \
                                imputed_times.get(args.impute_method, {})\
                                    .get(config_name, {}).get(scenario_key, {})\
                                        .get(train_size_str, {}).get(rand_idx, {})
                            Y_train_imputed = imputed_results.get("Y_train_imputed", None)
                            Y_val_imputed = imputed_results.get("Y_val_imputed", None)
                            Y_test_imputed = imputed_results.get("Y_test_imputed", None)
                        else:
                            Y_train_imputed = Y_val_imputed = Y_test_imputed = None

                        if Y_train_imputed is None:
                            survival_imputer = SurvivalEvalImputer(imputation_method=args.impute_method)
                            Y_train_imputed, Y_val_test_imputed = survival_imputer.fit_transform(Y_train, Y_val_test)
                            Y_val_imputed = Y_val_test_imputed[:val_size_]
                            Y_test_imputed = Y_val_test_imputed[val_size_:]

                        if args.meta_learner in ["t_learner", "x_learner"]:
                            if Y_train[W_train == 1, 1].sum() <= 1:
                                print(f"[Warning]: For {args.meta_learner}, No event in treatment group. Skipping iteration {rand_idx}.")
                                continue
                            if Y_train[W_train == 0, 1].sum() <= 1:
                                print(f"[Warning]: For {args.meta_learner}, No event in control group. Skipping iteration {rand_idx}.")
                                continue

                        learner.fit(X_train, W_train, Y_train_imputed)

                        end_time = time.time()
                        runtime = end_time - start_time
                        
                        # Save the model
                        learner.save_model(checkpoint_path)

                        start_time = time.time()
                        mse_val, cate_val_pred, ate_val_pred = learner.evaluate(X_val, cate_true_val, W_val)
                        mse_test, cate_test_pred, ate_test_pred = learner.evaluate(X_test, cate_true_test, W_test)
                        end_time = time.time()
                        inference_time = end_time - start_time

                        # Evaluate base regression models on test data
                        base_model_eval = learner.evaluate_test(X_test, Y_test_imputed, W_test)

                        results_dict[config_name][scenario_key][base_model][rand_idx] = {
                            "ate_true": ate_true,
                            "runtime": runtime,
                            # val set:
                            "cate_true_val": cate_true_val,
                            "cate_pred_val": cate_val_pred,
                            "ate_pred_val": ate_val_pred.mean_point,
                            "cate_mse_val": mse_val,
                            "ate_bias_val": ate_val_pred.mean_point - ate_true,
                            "ate_interval_val": ate_val_pred.conf_int_mean(),
                            "ate_statistics_val": ate_val_pred,
                            # test set:
                            "cate_true": cate_true_test,
                            "cate_pred": cate_test_pred,
                            "ate_pred": ate_test_pred.mean_point,
                            "cate_mse": mse_test,
                            "ate_bias": ate_test_pred.mean_point - ate_true,
                            "ate_interval": ate_test_pred.conf_int_mean(),
                            "ate_statistics": ate_test_pred,
                            "base_model_eval": base_model_eval, # Store base model evaluation results on test set
                        }

                        print(f'\ttraining time: {runtime:.0f} seconds; inference time: {inference_time:.0f} seconds')

                    with open(output_pickle_path, "wb") as f:
                        pickle.dump(results_dict, f)

                    # print(f"[Info]: Finished {config_name}, {scenario_key}, {base_model}, repeat {rand_idx}. CATE_MSE: {mse_test:.4f}, ATE_pred: {ate_test_pred.mean_point:.4f}, ATE_true: {ate_true:.4f}, ATE_bias: {ate_test_pred.mean_point - ate_true:.4f}")

                    # # loading (must use the correct subclass)
                    # loaded_learner = learner.load_model(checkpoint_path)
                    # loaded_mse_test, cate_test_pred, loaded_ate_test_pred = loaded_learner.evaluate(X_test, cate_true_test, W_test)
                    # print(f"[Loaded Info]: Finished {config_name}, {scenario_key}, {base_model}, repeat {rand_idx}. CATE_MSE: {loaded_mse_test:.4f}, ATE_pred: {loaded_ate_test_pred.mean_point:.4f}, ATE_true: {ate_true:.4f}, ATE_bias: {loaded_ate_test_pred.mean_point - ate_true:.4f}")

                if len(results_dict[config_name][scenario_key][base_model]) == 0:
                    print(f"[Warning]: No valid results for {config_name}, {scenario_key}, {base_model}. Skipping.")
                    continue

                # Save results to the setup dictionary
                avg = results_dict[config_name][scenario_key][base_model]
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
                results_dict[config_name][scenario_key][base_model]["average"] = {
                    # val set:
                    "mean_cate_mse_val": np.mean([avg[i]["cate_mse_val"] for i in range(num_repeats) if i in avg]),
                    "std_cate_mse_val": np.std([avg[i]["cate_mse_val"] for i in range(num_repeats) if i in avg]),
                    "mean_ate_pred_val": np.mean([avg[i]["ate_pred_val"] for i in range(num_repeats) if i in avg]),
                    "std_ate_pred_val": np.std([avg[i]["ate_pred_val"] for i in range(num_repeats) if i in avg]),
                    "mean_ate_bias_val": np.mean([avg[i]["ate_bias_val"] for i in range(num_repeats) if i in avg]),
                    "std_ate_bias_val": np.std([avg[i]["ate_bias_val"] for i in range(num_repeats) if i in avg]),
                    # test set:
                    "mean_cate_mse": np.mean([avg[i]["cate_mse"] for i in range(num_repeats) if i in avg]),
                    "std_cate_mse": np.std([avg[i]["cate_mse"] for i in range(num_repeats) if i in avg]),
                    "mean_ate_pred": np.mean([avg[i]["ate_pred"] for i in range(num_repeats) if i in avg]),
                    "std_ate_pred": np.std([avg[i]["ate_pred"] for i in range(num_repeats) if i in avg]),
                    "mean_ate_true": np.mean([avg[i]["ate_true"] for i in range(num_repeats) if i in avg]),
                    "std_ate_true": np.std([avg[i]["ate_true"] for i in range(num_repeats) if i in avg]),
                    "mean_ate_bias": np.mean([avg[i]["ate_bias"] for i in range(num_repeats) if i in avg]),
                    "std_ate_bias": np.std([avg[i]["ate_bias"] for i in range(num_repeats) if i in avg]),
                    "base_model_eval": base_model_eval_performance,

                    "runtime": np.mean([avg[i]["runtime"] for i in range(num_repeats) if i in avg]),
                }

            with open(output_pickle_path, "wb") as f:
                pickle.dump(results_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--dataset_name", type=str, default='synthetic')
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--result_dir", type=str, default='./results')
    parser.add_argument("--train_size", type=float, default=5000)
    parser.add_argument("--val_size", type=float, default=2500)
    parser.add_argument("--test_size", type=float, default=2500)
    parser.add_argument("--impute_method", type=str, default="Margin", choices=["Pseudo_obs", "Margin", "IPCW-T"])
    parser.add_argument("--meta_learner", type=str, default="t_learner", choices=["t_learner", "s_learner", "x_learner", "dr_learner"])
    parser.add_argument("--load_imputed", action="store_true")
    args = parser.parse_args()
    main(args)