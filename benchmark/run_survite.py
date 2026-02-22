"""
Run SurvITE experiments for SurvHTE-Benchmark
Based on run_causal_survival_forest.py
"""

import argparse
import os
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))
import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
from models_causal_survival.survite_model import SurvITEModel
from data import load_data, prepare_data_split
from models_utils.checkpoint_utils import get_checkpoint_path


def get_hyperparameters(dataset_name):
    """Get dataset-specific hyperparameters for SurvITE"""
    
    # Default hyperparameters (from original paper)
    default_params = {
        'z_dim': 32,
        'h_dim1': 32,
        'h_dim2': 32,
        'num_layers1': 3,
        'num_layers2': 2,
        'activation': 'relu',
        'dropout': 0.3,
        'ipm_type': 'wasserstein',
        'beta': 1e-3,
        'gamma': 0,  # Can be set to 1e-3 for smoothing
        'lr': 1e-3,
        # 'batch_size': 512,
        # 'epochs': 20000,
        'patience': 20
    }
    
    # Dataset-specific adjustments
    if dataset_name == 'synthetic':
        params = default_params.copy()
        # params['horizon'] = 30
        # params['epochs'] = 10000  # Smaller dataset, fewer epochs needed
        
    elif dataset_name == 'actg_syn':
        params = default_params.copy()
        # params['horizon'] = 30
        # params['h_dim1'] = 150  # Larger hidden dims for complex data
        # params['h_dim2'] = 150
        # params['batch_size'] = 256  # Smaller batch size for smaller dataset
        
    elif dataset_name == 'mimic_syn':
        params = default_params.copy()
        # params['horizon'] = 40
        params['h_dim1'] = 64
        params['h_dim2'] = 64
        # params['batch_size'] = 256
        
    elif dataset_name in ['twin']:
        params = default_params.copy()
        # params['horizon'] = 365
        # params['z_dim'] = 150  # Larger representation for complex data
        # params['h_dim1'] = 150
        # params['h_dim2'] = 150
        # params['num_layers1'] = 4  # Deeper networks for complex patterns
        # params['num_layers2'] = 3
        
    else:  # actg real data
        params = default_params.copy()
        # params['horizon'] = 30
        # params['h_dim1'] = 100
        # params['h_dim2'] = 100
        # params['batch_size'] = 256
    
    return params


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
    model_dir = args.model_dir
    save_models = args.save_models
    load_models = args.load_models
    train_size_str = f'{int(train_size*100)}%' if train_size<1 else f'{int(train_size)}'
    
    # Get dataset-specific hyperparameters
    hyperparams = get_hyperparameters(dataset_name)
    
    # Override with command line arguments if provided
    if args.ipm_type is not None:
        hyperparams['ipm_type'] = args.ipm_type
    if args.beta is not None:
        hyperparams['beta'] = args.beta
    if args.epochs is not None:
        hyperparams['epochs'] = args.epochs
    if args.batch_size is not None:
        hyperparams['batch_size'] = args.batch_size
    
    print(f"Using hyperparameters for {dataset_name}:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    # Load data
    experiment_setups, experiment_repeat_setups = load_data(
        dataset_name=dataset_name, 
        data_dir=data_dir
    )
    
    # Output path
    output_pickle_path = os.path.join(
        result_dir, 
        dataset_type, 
        f'models_causal_survival/survite/'
    )
    output_pickle_path += f"{dataset_name}_survite_repeats_{args.num_repeats}_train_{train_size_str}.pkl"
    os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
    print("Output results path:", output_pickle_path)
    
    # Load existing results if available
    if os.path.exists(output_pickle_path):
        print("Loading results from existing file.")
        with open(output_pickle_path, 'rb') as f:
            results_dict = pickle.load(f)
    else:
        print("Results file not found, creating new file.")
        results_dict = {}
    
    # Main experiment loop
    for config_name, setup_dict in tqdm(experiment_setups.items(), desc="Experiment Setups"):
        if config_name not in results_dict:
            results_dict[config_name] = {}
        
        for scenario_key in tqdm(setup_dict, desc=f"{config_name} Scenarios"):
            dataset_df = setup_dict[scenario_key]["dataset"]
            dataset_summary = setup_dict[scenario_key]["summary"]
            
            # Prepare data splits
            split_dict = prepare_data_split(
                dataset_df, experiment_repeat_setups, 
                num_repeats=num_repeats, 
                dataset_name=dataset_name,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                include_surv_probs=True if dataset_name in ['actg_syn', 'mimic_syn'] else False,
                include_rmst_med_horizon=True if dataset_name in ['actg_syn', 'mimic_syn'] else False
            )
            
            if scenario_key not in results_dict[config_name]:
                results_dict[config_name][scenario_key] = {}
            
            # Run experiments for each repeat
            for rand_idx in range(num_repeats):
                if dataset_name in ['synthetic', 'actg_real', 'twin']:
                    X_train, W_train, Y_train, cate_true_train = split_dict[rand_idx]['train']
                    X_val, W_val, Y_val, cate_true_val = split_dict[rand_idx]['val']
                    X_test, W_test, Y_test, cate_true_test = split_dict[rand_idx]['test']
                elif dataset_name in ['actg_syn', 'mimic_syn']:
                    X_train, W_train, Y_train, cate_true_train, cate_true_med_horizon_train, surv_probs_train = split_dict[rand_idx]['train']
                    X_val, W_val, Y_val, cate_true_val, cate_true_med_horizon_val, surv_probs_val = split_dict[rand_idx]['val']
                    X_test, W_test, Y_test, cate_true_test, cate_true_med_horizon_test, surv_probs_test = split_dict[rand_idx]['test']
                else:
                    raise ValueError(f"Unknown handling of dataset name: {dataset_name}")
                
                val_size_ = Y_val.shape[0]
                Y_val_test = np.vstack((Y_val, Y_test))
                
                max_time = Y_train[:, 0].max()
                ate_true = dataset_summary['ate']
                event_time_25pct = dataset_summary['event_time_25pct']
                event_time_50pct = dataset_summary['event_time_median']
                event_time_75pct = dataset_summary['event_time_75pct']
                
                # Model checkpoint path
                if save_models or load_models:
                    checkpoint_path = get_checkpoint_path(
                        model_dir=model_dir,
                        dataset_name=dataset_name,
                        dataset_type=dataset_type,
                        model_name='survite',
                        config_name=config_name,
                        scenario_key=scenario_key,
                        rand_idx=rand_idx,
                        train_size_str=train_size_str
                    )
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

                # TODO
                args.verbose = True
                
                # Initialize model
                learner = SurvITEModel(
                    seed=2025+rand_idx,
                    verbose=args.verbose,
                    **hyperparams
                )
                
                # Check if already computed
                if rand_idx in results_dict[config_name][scenario_key]:
                    if load_models and os.path.exists(checkpoint_path):
                        start_time = time.time()
                        learner.load_model(checkpoint_path)
                        print(f'Loaded model in {(time.time() - start_time):.0f} seconds')
                    runtime = results_dict[config_name][scenario_key][rand_idx]["runtime"]
                    print(f'\tTraining time from previous run: {runtime:.0f} seconds')
                    
                else:
                    # Train model
                    start_time = time.time()
                    history = learner.fit(X_train, W_train, Y_train)
                    end_time = time.time()
                    runtime = end_time - start_time
                    
                    # Save model if requested
                    if save_models:
                        learner.save_model(checkpoint_path)
                        print(f'Saved model to {checkpoint_path}')
                    
                    # Evaluate model
                    start_time = time.time()
                    mse_val, cate_val_pred, ate_val_pred = learner.evaluate(
                        X_val, cate_true_val, W_val
                    )
                    mse_test, cate_test_pred, ate_test_pred = learner.evaluate(
                        X_test, cate_true_test, W_test
                    )
                    end_time = time.time()
                    inference_time = end_time - start_time
                    
                    # Store results
                    results_dict[config_name][scenario_key][rand_idx] = {
                        "ate_true": ate_true,
                        "runtime": runtime,
                        "inference_time": inference_time,
                        # Validation set results
                        "cate_true_val": cate_true_val,
                        "cate_pred_val": cate_val_pred,
                        "ate_pred_val": ate_val_pred,
                        "cate_mse_val": mse_val,
                        "ate_bias_val": ate_val_pred - ate_true,
                        "ate_statistics_val": ate_val_pred, # these are the same as ate_pred for direct causal survival models
                        # Test set results
                        "cate_true": cate_true_test,
                        "cate_pred": cate_test_pred,
                        "ate_pred": ate_test_pred,
                        "cate_mse": mse_test,
                        "ate_bias": ate_test_pred - ate_true,
                        "ate_statistics": ate_test_pred, # these are the same as ate_pred for direct causal survival models
                    }

                    if dataset_name in ['actg_syn', 'mimic_syn']:

                        # Evaluate RMST at median horizon
                        mse_med_horizon_val, cate_med_horizon_val_pred, ate_med_horizon_val_pred = learner.evaluate_rmst_with_horizon(
                            X_val, cate_true_med_horizon_val, horizon=event_time_50pct, W=W_val
                        )
                        mse_med_horizon_test, cate_med_horizon_test_pred, ate_med_horizon_test_pred = learner.evaluate_rmst_with_horizon(
                            X_test, cate_true_med_horizon_test, horizon=event_time_50pct, W=W_test
                        )

                        ate_med_horizon_true = dataset_summary['ate_med_horizon']
                        results_dict[config_name][scenario_key][rand_idx]["ate_true_med_horizon"] = ate_med_horizon_true

                        results_dict[config_name][scenario_key][rand_idx]["cate_true_med_horizon_val"] = cate_true_med_horizon_val
                        results_dict[config_name][scenario_key][rand_idx]["cate_pred_med_horizon_val"] = cate_med_horizon_val_pred
                        results_dict[config_name][scenario_key][rand_idx]["ate_pred_med_horizon_val"] = ate_med_horizon_val_pred
                        results_dict[config_name][scenario_key][rand_idx]["cate_mse_med_horizon_val"] = mse_med_horizon_val
                        results_dict[config_name][scenario_key][rand_idx]["ate_med_horizon_bias_val"] = ate_med_horizon_val_pred - ate_med_horizon_true

                        results_dict[config_name][scenario_key][rand_idx]["cate_true_med_horizon"] = cate_true_med_horizon_test
                        results_dict[config_name][scenario_key][rand_idx]["cate_pred_med_horizon"] = cate_med_horizon_test_pred
                        results_dict[config_name][scenario_key][rand_idx]["ate_pred_med_horizon"] = ate_med_horizon_test_pred
                        results_dict[config_name][scenario_key][rand_idx]["cate_mse_med_horizon"] = mse_med_horizon_test
                        results_dict[config_name][scenario_key][rand_idx]["ate_bias_med_horizon"] = ate_med_horizon_test_pred - ate_med_horizon_true

                        # Evaluate survival probabilities
                        mse_p_surv_25pct_val, cate_p_surv_25pct_val_pred, ate_p_surv_25pct_val_pred = learner.evaluate_p_surv_with_horizon(
                            X_val, surv_probs_val[:,0], horizon=event_time_25pct, W=W_val
                        )
                        mse_p_surv_25pct_test, cate_p_surv_25pct_test_pred, ate_p_surv_25pct_test_pred = learner.evaluate_p_surv_with_horizon(
                            X_test, surv_probs_test[:,0], horizon=event_time_25pct, W=W_test
                        )
                        mse_p_surv_50pct_val, cate_p_surv_50pct_val_pred, ate_p_surv_50pct_val_pred = learner.evaluate_p_surv_with_horizon(
                            X_val, surv_probs_val[:,1], horizon=event_time_50pct, W=W_val
                        )
                        mse_p_surv_50pct_test, cate_p_surv_50pct_test_pred, ate_p_surv_50pct_test_pred = learner.evaluate_p_surv_with_horizon(
                            X_test, surv_probs_test[:,1], horizon=event_time_50pct, W=W_test
                        )
                        mse_p_surv_75pct_val, cate_p_surv_75pct_val_pred, ate_p_surv_75pct_val_pred = learner.evaluate_p_surv_with_horizon(
                            X_val, surv_probs_val[:,2], horizon=event_time_75pct, W=W_val
                        )
                        mse_p_surv_75pct_test, cate_p_surv_75pct_test_pred, ate_p_surv_75pct_test_pred = learner.evaluate_p_surv_with_horizon(
                            X_test, surv_probs_test[:,2], horizon=event_time_75pct, W=W_test
                        )

                        ate_p_surv_25pct_true = dataset_summary['ate_p_surv_t25']
                        ate_p_surv_50pct_true = dataset_summary['ate_p_surv_t50']
                        ate_p_surv_75pct_true = dataset_summary['ate_p_surv_t75']
                        results_dict[config_name][scenario_key][rand_idx]["ate_true_p_surv_25pct"] = ate_p_surv_25pct_true
                        results_dict[config_name][scenario_key][rand_idx]["ate_true_p_surv_50pct"] = ate_p_surv_50pct_true
                        results_dict[config_name][scenario_key][rand_idx]["ate_true_p_surv_75pct"] = ate_p_surv_75pct_true
                        
                        # Survival probability results at 25th percentile time
                        results_dict[config_name][scenario_key][rand_idx]["cate_true_p_surv_25pct_val"] = surv_probs_val[:,0]
                        results_dict[config_name][scenario_key][rand_idx]["cate_pred_p_surv_25pct_val"] = cate_p_surv_25pct_val_pred
                        results_dict[config_name][scenario_key][rand_idx]["ate_pred_p_surv_25pct_val"] = ate_p_surv_25pct_val_pred
                        results_dict[config_name][scenario_key][rand_idx]["cate_mse_p_surv_25pct_val"] = mse_p_surv_25pct_val
                        results_dict[config_name][scenario_key][rand_idx]["ate_bias_p_surv_25pct_val"] = ate_p_surv_25pct_val_pred - ate_p_surv_25pct_true
                        
                        results_dict[config_name][scenario_key][rand_idx]["cate_true_p_surv_25pct"] = surv_probs_test[:,0]
                        results_dict[config_name][scenario_key][rand_idx]["cate_pred_p_surv_25pct"] = cate_p_surv_25pct_test_pred
                        results_dict[config_name][scenario_key][rand_idx]["ate_pred_p_surv_25pct"] = ate_p_surv_25pct_test_pred
                        results_dict[config_name][scenario_key][rand_idx]["cate_mse_p_surv_25pct"] = mse_p_surv_25pct_test
                        results_dict[config_name][scenario_key][rand_idx]["ate_bias_p_surv_25pct"] = ate_p_surv_25pct_test_pred - ate_p_surv_25pct_true
                        
                        # Survival probability results at 50th percentile time
                        results_dict[config_name][scenario_key][rand_idx]["cate_true_p_surv_50pct_val"] = surv_probs_val[:,1]
                        results_dict[config_name][scenario_key][rand_idx]["cate_pred_p_surv_50pct_val"] = cate_p_surv_50pct_val_pred
                        results_dict[config_name][scenario_key][rand_idx]["ate_pred_p_surv_50pct_val"] = ate_p_surv_50pct_val_pred
                        results_dict[config_name][scenario_key][rand_idx]["cate_mse_p_surv_50pct_val"] = mse_p_surv_50pct_val
                        results_dict[config_name][scenario_key][rand_idx]["ate_bias_p_surv_50pct_val"] = ate_p_surv_50pct_val_pred - ate_p_surv_50pct_true
                        
                        results_dict[config_name][scenario_key][rand_idx]["cate_true_p_surv_50pct"] = surv_probs_test[:,1]
                        results_dict[config_name][scenario_key][rand_idx]["cate_pred_p_surv_50pct"] = cate_p_surv_50pct_test_pred
                        results_dict[config_name][scenario_key][rand_idx]["ate_pred_p_surv_50pct"] = ate_p_surv_50pct_test_pred
                        results_dict[config_name][scenario_key][rand_idx]["cate_mse_p_surv_50pct"] = mse_p_surv_50pct_test
                        results_dict[config_name][scenario_key][rand_idx]["ate_bias_p_surv_50pct"] = ate_p_surv_50pct_test_pred - ate_p_surv_50pct_true

                        # Survival probability results at 75th percentile time
                        results_dict[config_name][scenario_key][rand_idx]["cate_true_p_surv_75pct_val"] = surv_probs_val[:,2]
                        results_dict[config_name][scenario_key][rand_idx]["cate_pred_p_surv_75pct_val"] = cate_p_surv_75pct_val_pred
                        results_dict[config_name][scenario_key][rand_idx]["ate_pred_p_surv_75pct_val"] = ate_p_surv_75pct_val_pred
                        results_dict[config_name][scenario_key][rand_idx]["cate_mse_p_surv_75pct_val"] = mse_p_surv_75pct_val
                        results_dict[config_name][scenario_key][rand_idx]["ate_bias_p_surv_75pct_val"] = ate_p_surv_75pct_val_pred - ate_p_surv_75pct_true
                        
                        results_dict[config_name][scenario_key][rand_idx]["cate_true_p_surv_75pct"] = surv_probs_test[:,2]
                        results_dict[config_name][scenario_key][rand_idx]["cate_pred_p_surv_75pct"] = cate_p_surv_75pct_test_pred
                        results_dict[config_name][scenario_key][rand_idx]["ate_pred_p_surv_75pct"] = ate_p_surv_75pct_test_pred
                        results_dict[config_name][scenario_key][rand_idx]["cate_mse_p_surv_75pct"] = mse_p_surv_75pct_test
                        results_dict[config_name][scenario_key][rand_idx]["ate_bias_p_surv_75pct"] = ate_p_surv_75pct_test_pred - ate_p_surv_75pct_true

                    print(f'\tTraining time: {runtime:.0f}s; Inference time: {inference_time:.0f}s')
                    print(f'\tValidation CATE RMSE: {np.sqrt(mse_val):.4f}, Test CATE RMSE: {np.sqrt(mse_test):.4f}')
                    print(f'\tATE True: {ate_true:.4f}, ATE Pred: {ate_test_pred:.4f}')
                    
                    # Save results
                    with open(output_pickle_path, "wb") as f:
                        pickle.dump(results_dict, f)
            
            # Compute aggregate statistics
            if len(results_dict[config_name][scenario_key]) == 0:
                print(f"[Warning]: No valid results for {config_name}, {scenario_key}. Skipping.")
                continue
            
            avg = results_dict[config_name][scenario_key]
            results_dict[config_name][scenario_key]["average"] = {
                # Validation set
                "mean_cate_mse_val": np.mean([avg[i]["cate_mse_val"] for i in range(num_repeats) if i in avg]),
                "std_cate_mse_val": np.std([avg[i]["cate_mse_val"] for i in range(num_repeats) if i in avg]),
                "mean_ate_pred_val": np.mean([avg[i]["ate_pred_val"] for i in range(num_repeats) if i in avg]),
                "std_ate_pred_val": np.std([avg[i]["ate_pred_val"] for i in range(num_repeats) if i in avg]),
                "mean_ate_bias_val": np.mean([avg[i]["ate_bias_val"] for i in range(num_repeats) if i in avg]),
                "std_ate_bias_val": np.std([avg[i]["ate_bias_val"] for i in range(num_repeats) if i in avg]),
                # Test set
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

            if dataset_name in ['actg_syn', 'mimic_syn']:
                # Validation set RMST at median horizon
                results_dict[config_name][scenario_key]["average"]["mean_cate_mse_med_horizon_val"] = np.mean(
                    [avg[i]["cate_mse_med_horizon_val"] for i in range(num_repeats) if i in avg]
                )
                results_dict[config_name][scenario_key]["average"]["std_cate_mse_med_horizon_val"] = np.std(
                    [avg[i]["cate_mse_med_horizon_val"] for i in range(num_repeats) if i in avg]
                )
                results_dict[config_name][scenario_key]["average"]["mean_ate_bias_med_horizon_val"] = np.mean(
                    [avg[i]["ate_med_horizon_bias_val"] for i in range(num_repeats) if i in avg]
                )
                results_dict[config_name][scenario_key]["average"]["std_ate_bias_med_horizon_val"] = np.std(
                    [avg[i]["ate_med_horizon_bias_val"] for i in range(num_repeats) if i in avg]
                )
                # Test set RMST at median horizon
                results_dict[config_name][scenario_key]["average"]["mean_cate_mse_med_horizon"] = np.mean(
                    [avg[i]["cate_mse_med_horizon"] for i in range(num_repeats) if i in avg]
                )
                results_dict[config_name][scenario_key]["average"]["std_cate_mse_med_horizon"] = np.std(
                    [avg[i]["cate_mse_med_horizon"] for i in range(num_repeats) if i in avg]
                )
                results_dict[config_name][scenario_key]["average"]["mean_ate_bias_med_horizon"] = np.mean(
                    [avg[i]["ate_bias_med_horizon"] for i in range(num_repeats) if i in avg]
                )
                results_dict[config_name][scenario_key]["average"]["std_ate_bias_med_horizon"] = np.std(
                    [avg[i]["ate_bias_med_horizon"] for i in range(num_repeats) if i in avg]
                )

                # Survival probabilities at 25th, 50th, 75th percentiles
                for perc in ['25pct', '50pct', '75pct']:
                    # Validation set
                    results_dict[config_name][scenario_key]["average"][f"mean_cate_mse_p_surv_{perc}_val"] = np.mean(
                        [avg[i][f"cate_mse_p_surv_{perc}_val"] for i in range(num_repeats) if i in avg]
                    )
                    results_dict[config_name][scenario_key]["average"][f"std_cate_mse_p_surv_{perc}_val"] = np.std(
                        [avg[i][f"cate_mse_p_surv_{perc}_val"] for i in range(num_repeats) if i in avg]
                    )
                    results_dict[config_name][scenario_key]["average"][f"mean_ate_bias_p_surv_{perc}_val"] = np.mean(
                        [avg[i][f"ate_bias_p_surv_{perc}_val"] for i in range(num_repeats) if i in avg]
                    )
                    results_dict[config_name][scenario_key]["average"][f"std_ate_bias_p_surv_{perc}_val"] = np.std(
                        [avg[i][f"ate_bias_p_surv_{perc}_val"] for i in range(num_repeats) if i in avg]
                    )
                    # Test set
                    results_dict[config_name][scenario_key]["average"][f"mean_cate_mse_p_surv_{perc}"] = np.mean(
                        [avg[i][f"cate_mse_p_surv_{perc}"] for i in range(num_repeats) if i in avg]
                    )
                    results_dict[config_name][scenario_key]["average"][f"std_cate_mse_p_surv_{perc}"] = np.std(
                        [avg[i][f"cate_mse_p_surv_{perc}"] for i in range(num_repeats) if i in avg]
                    )
                    results_dict[config_name][scenario_key]["average"][f"mean_ate_bias_p_surv_{perc}"] = np.mean(
                        [avg[i][f"ate_bias_p_surv_{perc}"] for i in range(num_repeats) if i in avg]
                    )
                    results_dict[config_name][scenario_key]["average"][f"std_ate_bias_p_surv_{perc}"] = np.std(
                        [avg[i][f"ate_bias_p_surv_{perc}"] for i in range(num_repeats) if i in avg]
                    )
            
            # Save updated results
            with open(output_pickle_path, "wb") as f:
                pickle.dump(results_dict, f)
            
            print(f"Completed {config_name}, {scenario_key}:")
            print(f"  Mean CATE RMSE: {np.sqrt(results_dict[config_name][scenario_key]['average']['mean_cate_mse']):.4f} "
                  f"± {np.sqrt(results_dict[config_name][scenario_key]['average']['std_cate_mse']):.4f}")
            print(f"  Mean ATE Bias: {results_dict[config_name][scenario_key]['average']['mean_ate_bias']:.4f} "
                  f"± {results_dict[config_name][scenario_key]['average']['std_ate_bias']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--dataset_name", type=str, default='synthetic',
                        choices=['synthetic', 'actg_syn', 'mimic_syn', 'actg', 'twin'],
                        help='Dataset name for the experiment')
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--result_dir", type=str, default='./results')
    parser.add_argument("--train_size", type=float, default=5000)
    parser.add_argument("--val_size", type=float, default=2500)
    parser.add_argument("--test_size", type=float, default=2500)
    
    # Model save/load arguments
    parser.add_argument("--model_dir", type=str, default='./models')
    parser.add_argument("--save_models", action='store_true', help='Save trained models')
    parser.add_argument("--load_models", action='store_true', help='Load pre-trained models')
    
    # SurvITE-specific hyperparameters (optional overrides)
    parser.add_argument("--ipm_type", type=str, default="wasserstein", 
                       choices=['wasserstein', 'mmd', 'no_ipm'],
                       help='IPM type for domain adaptation (default of SurvITE paper: wasserstein)')
    parser.add_argument("--beta", type=float, default=0.001,
                       help='IPM regularization weight (default: 0.001)')
    parser.add_argument("--epochs", type=int, default=1500,
                       help='Maximum training epochs')
    parser.add_argument("--batch_size", type=int, default=256,
                       help='Training batch size')
    parser.add_argument("--verbose", action='store_true',
                       help='Print training progress')
    
    args = parser.parse_args()
    main(args)