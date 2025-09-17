import argparse
import os
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))
import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm, trange
from models_causal_impute.survival_eval_impute import SurvivalEvalImputer
from data import load_data, prepare_data_split


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

    experiment_setups, experiment_repeat_setups = load_data(dataset_name=dataset_name, data_dir=data_dir)

    output_pickle_path = os.path.join(data_dir, dataset_type, f'imputed_times_lookup_{dataset_name}.pkl')
    print(f'output file: {output_pickle_path}')
    imputation_methods_list = ['Pseudo_obs', 'Margin', 'IPCW-T']

    if os.path.exists(output_pickle_path):
        print("Loading imputation times from existing file.")
        with open(output_pickle_path, 'rb') as f:
            imputed_times = pickle.load(f)
    else:
        print("Imputation times not found, creating new file.")
        imputed_times = {}

    start_time = end_time = 0

    for imputation_method_idx in trange(len(imputation_methods_list), desc="Imputation Methods"):
        imputation_method = imputation_methods_list[imputation_method_idx]

        if imputed_times.get(imputation_method) is None:
            print(f"Imputation times not found for {imputation_method}, creating new entry.")
            imputed_times[imputation_method] = {}

        # for config_name, setup_dict in tqdm(experiment_setups.items(), desc="Experiment Setups"):
        for config_name, setup_dict in experiment_setups.items():

            # Check if imputed_times[imputation_method] has the config_name
            if config_name not in imputed_times[imputation_method]:
                print(f"Creating new entry for '{config_name}' in imputed times['{imputation_method}'].")
                imputed_times[imputation_method][config_name] = {}

            for scenario_key in setup_dict:
                dataset_df = setup_dict[scenario_key]["dataset"]
                split_dict = prepare_data_split(
                    dataset_df, experiment_repeat_setups, 
                    num_repeats=num_repeats, 
                    dataset_name=dataset_name,
                    train_size=train_size,
                    val_size=val_size,
                    test_size=test_size
                )

                # check if imputed_times[imputation_method][config_name] has the scenario_key
                if scenario_key not in imputed_times[imputation_method][config_name]:
                    print(f"Creating new entry for '{scenario_key}' in imputed times['{imputation_method}']['{config_name}'].")
                    imputed_times[imputation_method][config_name][scenario_key] = {}

                train_size_str = f'{int(train_size*100)}%' if train_size<1 else f'{int(train_size)}'

                # check if imputed_times[imputation_method][config_name][scenario_key] has the num_training_data_points
                if train_size_str not in imputed_times[imputation_method][config_name][scenario_key]:
                    # print(f"Creating new entry for '{train_size_str}' in imputed times['{imputation_method}']['{config_name}']['{scenario_key}'].")
                    imputed_times[imputation_method][config_name][scenario_key][train_size_str] = {}

                for rand_idx in range(num_repeats):
                    X_train, W_train, Y_train, cate_true_train = split_dict[rand_idx]['train']
                    X_val, W_val, Y_val, cate_true_val = split_dict[rand_idx]['val']
                    X_test, W_test, Y_test, cate_true_test = split_dict[rand_idx]['test']
                    val_size_ = Y_val.shape[0]
                    Y_val_test = np.vstack((Y_val, Y_test))

                    # Check if imputed_times[imputation_method][config_name][scenario_key][train_size_str] has the rand_idx
                    if rand_idx not in imputed_times[imputation_method][config_name][scenario_key][train_size_str]:
                        # print(f"Creating new entry for '{rand_idx}' in imputed times['{imputation_method}']['{config_name}']['{scenario_key}']['{train_size_str}'].")
                        imputed_times[imputation_method][config_name][scenario_key][train_size_str][rand_idx] = {}

                        start_time = time.time()

                        # impute the missing values
                        survival_imputer = SurvivalEvalImputer(imputation_method=imputation_method, verbose=False)
                        Y_train_imputed, Y_val_test_imputed = survival_imputer.fit_transform(Y_train, Y_val_test)
                        y_val_imputed = Y_val_test_imputed[:val_size_]
                        y_test_imputed = Y_val_test_imputed[val_size_:]

                        end_time = time.time()

                        imputed_times[imputation_method][config_name][scenario_key][train_size_str][rand_idx] = {
                            "Y_train_imputed": Y_train_imputed,
                            "Y_val_imputed": y_val_imputed,
                            "Y_test_imputed": y_test_imputed,
                            "runtime": end_time - start_time
                        }
                        print(f"\t'{imputation_method}' Imputation completed for '{config_name}', '{scenario_key}', " +
                            f"num_training: {train_size_str}, repeat {rand_idx} in {end_time - start_time:.0f} seconds.")
                            
                        # Save progress to disk
                        with open(output_pickle_path, "wb") as f:
                            pickle.dump(imputed_times, f)
                    elif rand_idx in imputed_times[imputation_method][config_name][scenario_key][train_size_str] and \
                        "runtime" in imputed_times[imputation_method][config_name][scenario_key][train_size_str][rand_idx]:
                        runtime = imputed_times[imputation_method][config_name][scenario_key][train_size_str][rand_idx]["runtime"]
                        print(f"\tFound existing entry for repeat {rand_idx} in imputed times['{imputation_method}']['{config_name}']['{scenario_key}'][{train_size_str}]," ,
                              f"time {runtime:.0f} seconds")

                    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--dataset_name", type=str, default='synthetic')
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--train_size", type=float, default=5000)
    parser.add_argument("--val_size", type=float, default=2500)
    parser.add_argument("--test_size", type=float, default=2500)
    args = parser.parse_args()
    main(args)