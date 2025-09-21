import argparse
import os
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))
import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
from models_causal_survival.causal_survival_forest import CausalSurvivalForestGRF
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

    if dataset_name == 'synthetic':
        failure_times_grid_size = 500
        horizon, min_node_size = None, None # use default
    elif dataset_name == 'actg_syn':
        failure_times_grid_size = 200
        horizon, min_node_size = 30, 18
    elif dataset_name == 'mimic_syn':
        failure_times_grid_size = 200
        horizon, min_node_size = 40, 18
    elif dataset_name in ['twin30', 'twin180']:
        print('Use non-uniform discretization for twin datasets.')
        failure_times_grid_size = 200
        horizon, min_node_size = 365, 18
        # failure_times_grid: non-uniform discretization for twin datasets
        # i.e. resolution of days in the first 30 days and months after the first 30 days
        # every day for 1 month, then every month
        failure_times_grid = np.concatenate([np.arange(0, 30), np.arange(30, 365, 30)]) 
    else: # actg real data
        failure_times_grid_size = 200
        horizon, min_node_size = 30, 18


    experiment_setups, experiment_repeat_setups = load_data(dataset_name=dataset_name, data_dir=data_dir)
    
    output_pickle_path = os.path.join(result_dir, dataset_type, f'models_causal_survival/causal_survival_forest/')
    output_pickle_path += f"{dataset_name}_causal_survival_forest_repeats_{args.num_repeats}_train_{train_size_str}.pkl"
    os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
    print("Output results path:", output_pickle_path)


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


                
                if dataset_name == 'synthetic':
                    learner = CausalSurvivalForestGRF(failure_times_grid_size=failure_times_grid_size,
                                                    seed=2025+rand_idx)
                else:
                    learner = CausalSurvivalForestGRF(failure_times_grid_size=failure_times_grid_size,
                                                    horizon=horizon, min_node_size=min_node_size, 
                                                    seed=2025+rand_idx)
                

                if rand_idx in results_dict[config_name][scenario_key]:
                    # t_ = time.time()
                    # learner.load_model(checkpoint_path)
                    # print(f'Took {(time.time() - t_):.0f} seconds to load the model')
                    runtime = results_dict[config_name][scenario_key][rand_idx]["runtime"]
                    print(f'\ttraining time from previous run: {runtime:.0f} seconds')

                else:
                    start_time = time.time()

                    if dataset_name in ['twin30', 'twin180']:
                        learner.fit(X_train, W_train, Y_train, failure_times_grid=failure_times_grid)
                    else:
                        learner.fit(X_train, W_train, Y_train)

                    end_time = time.time()
                    runtime = end_time - start_time

                    start_time = time.time()
                    mse_val, cate_val_pred, ate_val_pred = learner.evaluate(X_val, cate_true_val, W_val)
                    mse_test, cate_test_pred, ate_test_pred = learner.evaluate(X_test, cate_true_test, W_test)
                    end_time = time.time()
                    inference_time = end_time - start_time

                    results_dict[config_name][scenario_key][rand_idx] = {
                        "ate_true": ate_true,
                        "runtime": runtime,
                        "inference_time": inference_time,
                        # val set:
                        "cate_true_val": cate_true_val,
                        "cate_pred_val": cate_val_pred,
                        "ate_pred_val": ate_val_pred,
                        "cate_mse_val": mse_val,
                        "ate_bias_val": ate_val_pred - ate_true,
                        "ate_statistics_val": ate_val_pred,
                        # test set:
                        "cate_true": cate_true_test,
                        "cate_pred": cate_test_pred,
                        "ate_pred": ate_test_pred,
                        "cate_mse": mse_test,
                        "ate_bias": ate_test_pred - ate_true,
                        "ate_statistics": ate_test_pred,
                    }

                    print(f'\ttraining time: {runtime:.0f} seconds; inference time: {inference_time:.0f} seconds')

                    with open(output_pickle_path, "wb") as f:
                        pickle.dump(results_dict, f)

                    # print(f"Completed {config_name}, {scenario_key}, repeat {rand_idx}: CATE MSE={mse_test:.4f}, ATE True={ate_true:.4f}, ATE Pred={ate_test_pred.mean_point:.4f}")
            

            if len(results_dict[config_name][scenario_key]) == 0:
                print(f"[Warning]: No valid results for {config_name}, {scenario_key}. Skipping.")
                continue

            # Save results to the setup dictionary
            avg = results_dict[config_name][scenario_key]
            results_dict[config_name][scenario_key]["average"] = {
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
                "mean_ate_bias": np.mean([avg[i]["ate_bias"] for i in range(num_repeats) if i in avg]),
                "std_ate_bias": np.std([avg[i]["ate_bias"] for i in range(num_repeats) if i in avg]),

                "mean_ate_true": np.mean([avg[i]["ate_true"] for i in range(num_repeats) if i in avg]),
                "std_ate_true": np.std([avg[i]["ate_true"] for i in range(num_repeats) if i in avg]),
                "runtime": np.mean([avg[i]["runtime"] for i in range(num_repeats) if i in avg])
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
    # We do not provide model saving/loading for CSF as it is from an R module
    args = parser.parse_args()
    main(args)