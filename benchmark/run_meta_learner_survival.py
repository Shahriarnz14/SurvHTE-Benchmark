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
    num_repeats = args.num_repeats
    dataset_name = args.dataset_name
    dataset_type = (
        "synthetic" if dataset_name == "synthetic"
        else "semi-synthetic" if dataset_name in ["mimic_syn", "mimic_syn_2", "actg_syn"]
        else "real"
    )
    if args.include_med_rmst and args.survival_metric != "mean":
        print("[Warning]: include_med_rmst only makes sense with survival_metric='mean' (RMST). Skipping med-RMST eval.")
        args.include_med_rmst = False
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    data_dir = args.data_dir
    result_dir = args.result_dir
    train_size_str = f'{int(train_size*100)}%' if train_size<1 else f'{int(train_size)}'

    experiment_setups, experiment_repeat_setups = load_data(dataset_name=dataset_name, data_dir=data_dir)

    output_pickle_path = os.path.join(result_dir, dataset_type, f"models_causal_survival_meta/{args.meta_learner}/")
    output_pickle_path += f"{dataset_name}_{args.meta_learner}_{args.base_survival_model}_repeats_{args.num_repeats}_train_{train_size_str}.pkl"
    if args.exp_name != "":
        output_pickle_path = output_pickle_path.replace(".pkl", f"_{args.exp_name}.pkl")
    os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
    print("Output results path:", output_pickle_path)

    # Define base survival models to use
    base_model = args.base_survival_model
    if os.path.exists(output_pickle_path):
        print(f"Pickle file already exists. Loading from {output_pickle_path}...")
        with open(output_pickle_path, "rb") as f:
            results_dict = pickle.load(f)
    else:
        print("Results file not found, creating new file.")
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
        # if config_name in results_dict:
        #     print(f"Skipping setup {config_name} as it already exists in results.")
        #     continue
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
                test_size=test_size,
                include_surv_probs=args.include_surv_probs,
                include_rmst_med_horizon=args.include_med_rmst
            )
            if scenario_key not in results_dict[config_name]:
                results_dict[config_name][scenario_key] = {}


            for rand_idx in range(num_repeats):
                # X_train, W_train, Y_train, cate_true_train = split_dict[rand_idx]['train']
                # X_val, W_val, Y_val, cate_true_val = split_dict[rand_idx]['val']
                # X_test, W_test, Y_test, cate_true_test = split_dict[rand_idx]['test']
                train_tuple = split_dict[rand_idx]['train']
                val_tuple = split_dict[rand_idx]['val']
                test_tuple = split_dict[rand_idx]['test']
                def unpack(tup):
                    if args.include_med_rmst and args.include_surv_probs:
                        X, W, Y, cate, cate_med, cate_surv = tup
                    elif args.include_med_rmst:
                        X, W, Y, cate, cate_med = tup
                        cate_surv = None
                    elif args.include_surv_probs:
                        X, W, Y, cate, cate_surv = tup
                        cate_med = None
                    else:
                        X, W, Y, cate = tup
                        cate_med = None
                        cate_surv = None
                    return X, W, Y, cate, cate_med, cate_surv

                X_train, W_train, Y_train, cate_true_train, cate_true_med_horizon_train, cate_true_surv_train = unpack(train_tuple)
                X_val,   W_val,   Y_val,   cate_true_val,   cate_true_med_horizon_val,   cate_true_surv_val   = unpack(val_tuple)
                X_test,  W_test,  Y_test,  cate_true_test,  cate_true_med_horizon_test,  cate_true_surv_test  = unpack(test_tuple)

                val_size_ = Y_val.shape[0]
                Y_val_test = np.vstack((Y_val, Y_test))
                
                max_time = Y_train[:, 0].max()
                ate_true = dataset_summary['ate']
                event_time_25pct = dataset_summary['event_time_25pct']
                event_time_50pct = dataset_summary['event_time_median']
                event_time_75pct = dataset_summary['event_time_75pct']

                if args.max_time == 0.5:
                    max_time = event_time_50pct
                
                # Generate checkpoint path
                if args.save_model:
                    checkpoint_path = get_checkpoint_path(
                        dataset_type=dataset_type,
                        causal_config=config_name,
                        scenario=scenario_key,
                        model_family=args.meta_learner,
                        model_name=f"{args.meta_learner}_{base_model}_{args.survival_metric}",
                        repeat_idx=rand_idx
                    )
                
                # Initialize the appropriate meta-learner
                if args.meta_learner == "t_learner_survival":
                    learner = TLearnerSurvival(
                        base_model_name=base_model,
                        base_model_grid=hyperparameter_grids,
                        metric=args.survival_metric,
                        med_time=event_time_50pct,
                        max_time=max_time
                    )
                elif args.meta_learner == "s_learner_survival":
                    learner = SLearnerSurvival(
                        base_model_name=base_model,
                        base_model_grid=hyperparameter_grids,
                        metric=args.survival_metric,
                        med_time=event_time_50pct,
                        max_time=max_time
                    )
                elif args.meta_learner == "matching_learner_survival":
                    learner = MatchingLearnerSurvival(
                        base_model_name=base_model,
                        base_model_grid=hyperparameter_grids,
                        metric=args.survival_metric,
                        num_matches=args.num_matches,
                        med_time=event_time_50pct,
                        max_time=max_time
                    )

                if rand_idx in results_dict[config_name][scenario_key]:
                    runtime = results_dict[config_name][scenario_key][rand_idx]["runtime"]
                    print(f'\ttraining time from previous run: {runtime:.0f} seconds')

                else:
                    start_time = time.time()

                    if args.meta_learner == "t_learner_survival":
                        if Y_train[W_train == 1, 1].sum() <= 1:
                            print(f"[Warning]: For {args.meta_learner}, No event in treatment group. Skipping iteration {rand_idx}.")
                            continue
                        if Y_train[W_train == 0, 1].sum() <= 1:
                            print(f"[Warning]: For {args.meta_learner}, No event in control group. Skipping iteration {rand_idx}.")
                            continue

                    # Fit the learner
                    learner.fit(X_train, W_train, Y_train)

                    end_time = time.time()
                    runtime = end_time - start_time


                    # Save the model
                    ## The model random survival forest is too large (usually several GBs!) when saved, 
                    ## so by default we do not save trained models for it.
                    ## Trained DeepSurv/DeepHit models will be saved by default though.
                    if args.save_model:
                        t_ = time.time()
                        learner.save_model(checkpoint_path)
                        print(f'Took {(time.time() - t_):.0f} seconds to save the model')
                    
                    start_time = time.time()
                    # Evaluate base survival models
                    base_model_eval = learner.evaluate_test(X_test, Y_test, W_test)
                    base_model_eval_val = learner.evaluate_test(X_val, Y_val, W_val)
                    
                    # Evaluate causal effect predictions
                    mse_test, cate_test_pred, ate_test_pred = learner.evaluate(X_test, cate_true_test, W_test)
                    mse_val, cate_val_pred, ate_val_pred = learner.evaluate(X_val, cate_true_val, W_val)

                    if args.include_med_rmst:
                        # Evaluate RMST at median horizon
                        mse_med_horizon_val, cate_med_horizon_val_pred, ate_med_horizon_val_pred = learner.evaluate_rmst_with_horizon(
                            X_val, cate_true_med_horizon_val, horizon=event_time_50pct, W=W_val
                        )
                        mse_med_horizon_test, cate_med_horizon_test_pred, ate_med_horizon_test_pred = learner.evaluate_rmst_with_horizon(
                            X_test, cate_true_med_horizon_test, horizon=event_time_50pct, W=W_test
                        )

                    # Evaluate CATE based on survival probabilities at 25/50/75 percentiles
                    cate_surv_test = None
                    cate_surv_val = None
                    mse_surv_test = None
                    mse_surv_val = None

                    if args.include_surv_probs:
                        horizons = np.array(
                            [event_time_25pct, event_time_50pct, event_time_75pct],
                            dtype=float,
                        )

                        # Predicted CATE for survival probability (test & val)
                        cate_time_test_surv, cate_surv_test = learner.predict_cate_surv_probs(
                            X_test, horizons=horizons, W=W_test
                        )
                        cate_time_val_surv, cate_surv_val = learner.predict_cate_surv_probs(
                            X_val, horizons=horizons, W=W_val
                        )

                        # Using ground-truth survival prob CATE, compute MSE per horizon
                        if (cate_true_surv_test is not None) and (cate_surv_test is not None):
                            # shape: (n_horizons,)
                            mse_surv_test = np.mean(
                                (cate_true_surv_test - cate_surv_test) ** 2, axis=0
                            )
                        if (cate_true_surv_val is not None) and (cate_surv_val is not None):
                            mse_surv_val = np.mean(
                                (cate_true_surv_val - cate_surv_val) ** 2, axis=0
                            )


                    end_time = time.time()
                    inference_time = end_time - start_time

                    entry = {
                        "ate_true": ate_true,
                        "runtime": runtime,
                        "inference_time": inference_time,
                        # val set:
                        "cate_true_val": cate_true_val,
                        "cate_pred": cate_val_pred,
                        "ate_pred_val": ate_val_pred,
                        "cate_mse_val": mse_val,
                        "ate_bias_val": ate_val_pred - ate_true,
                        "ate_statistics_val": ate_val_pred,
                        "base_model_eval_val": base_model_eval_val,  # Store base model evaluation results
                        # test set:
                        "cate_true": cate_true_test,
                        "cate_pred": cate_test_pred,
                        "ate_pred": ate_test_pred,
                        "cate_mse": mse_test,
                        "ate_bias": ate_test_pred - ate_true,
                        "ate_statistics": ate_test_pred,
                        "base_model_eval": base_model_eval,  # Store base model evaluation results
                    }

                    if args.include_med_rmst:
                        ate_med_horizon_true = dataset_summary.get('ate_med_horizon', None)
                        if ate_med_horizon_true is None:
                            time_at_med = dataset_df['T'].quantile(0.5)
                            ate_med_horizon_true = (
                                np.minimum(dataset_df['T1'], time_at_med)
                                - np.minimum(dataset_df['T0'], time_at_med)
                            ).mean()
                        entry.update({
                        "ate_true_med_horizon": ate_med_horizon_true,
                        # val set:
                        "cate_true_med_horizon_val": cate_true_med_horizon_val,
                        "cate_pred_med_horizon_val": cate_med_horizon_val_pred,
                        "ate_pred_med_horizon_val": ate_med_horizon_val_pred,
                        "cate_mse_med_horizon_val": mse_med_horizon_val,
                        "ate_med_horizon_bias_val": ate_med_horizon_val_pred - ate_med_horizon_true,
                        # test set:
                        "cate_true_med_horizon": cate_true_med_horizon_test,
                        "cate_pred_med_horizon": cate_med_horizon_test_pred,
                        "ate_pred_med_horizon": ate_med_horizon_test_pred,
                        "cate_mse_med_horizon": mse_med_horizon_test,
                        "ate_bias_med_horizon": ate_med_horizon_test_pred - ate_med_horizon_true,
                        })

                    if args.include_surv_probs and (cate_surv_test is not None) and (cate_surv_val is not None):
                        entry.update({
                            "surv_horizons": np.array(
                                [event_time_25pct, event_time_50pct, event_time_75pct],
                                dtype=float,
                            ),
                            # ground-truth survival CATE (if provided)
                            "cate_surv_true_val": cate_true_surv_val,
                            "cate_surv_true": cate_true_surv_test,
                            # predicted survival CATE
                            "cate_surv_pred_val": cate_surv_val,   # shape (n_val, 3)
                            "cate_surv_pred": cate_surv_test,       # shape (n_test, 3)
                            # MSE per horizon
                            "cate_surv_mse_val": mse_surv_val,      # shape (3,) or None
                            "cate_surv_mse": mse_surv_test,         # shape (3,) or None
                        })

                    results_dict[config_name][scenario_key][rand_idx] = entry

                    print(f'\ttraining time: {runtime:.0f} seconds; inference time: {inference_time:.0f} seconds')

                    with open(output_pickle_path, "wb") as f:
                        pickle.dump(results_dict, f)

                # print(f"Completed {config_name}, {scenario_key}, repeat {rand_idx}: CATE MSE={mse_test:.4f}, ATE True={ate_true:.4f}, ATE Pred={ate_test_pred:.4f}")
                # loaded_learner = learner.load_model(checkpoint_path)
                # loaded_mse_test, cate_test_pred, loaded_ate_test_pred = loaded_learner.evaluate(X_test, cate_true_test, W_test)
                # print(f"Loaded model evaluation: CATE MSE={loaded_mse_test:.4f}, ATE True={ate_true:.4f}, ATE Pred={loaded_ate_test_pred:.4f}")

                # import pdb; pdb.set_trace()

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
                
            avg_dict = {
                # val set:
                "mean_cate_mse_val": np.mean([avg[i]["cate_mse_val"] for i in range(num_repeats) if i in avg]),
                "std_cate_mse_val": np.std([avg[i]["cate_mse_val"] for i in range(num_repeats) if i in avg]),
                "mean_ate_pred_val": np.mean([avg[i]["ate_pred_val"] for i in range(num_repeats) if i in avg]),
                "std_ate_pred_val": np.std([avg[i]["ate_pred_val"] for i in range(num_repeats) if i in avg]),
                "mean_ate_bias_val": np.mean([avg[i]["ate_bias_val"] for i in range(num_repeats) if i in avg]),
                "std_ate_bias_val": np.std([avg[i]["ate_bias_val"] for i in range(num_repeats) if i in avg]),
                "base_model_eval_val" : base_model_eval_performance_val,
                # test set:
                "mean_cate_mse": np.mean([avg[i]["cate_mse"] for i in range(num_repeats) if i in avg]),
                "std_cate_mse": np.std([avg[i]["cate_mse"] for i in range(num_repeats) if i in avg]),
                "mean_ate_pred": np.mean([avg[i]["ate_pred"] for i in range(num_repeats) if i in avg]),
                "std_ate_pred": np.std([avg[i]["ate_pred"] for i in range(num_repeats) if i in avg]),
                "mean_ate_bias": np.mean([avg[i]["ate_bias"] for i in range(num_repeats) if i in avg]),
                "std_ate_bias": np.std([avg[i]["ate_bias"] for i in range(num_repeats) if i in avg]),
                "base_model_eval" : base_model_eval_performance,

                "mean_ate_true": np.mean([avg[i]["ate_true"] for i in range(num_repeats) if i in avg]),
                "std_ate_true": np.std([avg[i]["ate_true"] for i in range(num_repeats) if i in avg]),
                "runtime": np.mean([avg[i]["runtime"] for i in range(num_repeats) if i in avg]),
                }
            
            if args.include_med_rmst:
                avg_dict.update({
                    # val
                    "mean_cate_mse_med_horizon_val": np.mean([avg[i]["cate_mse_med_horizon_val"] for i in range(num_repeats) if i in avg]),
                    "std_cate_mse_med_horizon_val":  np.std( [avg[i]["cate_mse_med_horizon_val"] for i in range(num_repeats) if i in avg]),
                    "mean_ate_pred_med_horizon_val": np.mean([avg[i]["ate_pred_med_horizon_val"] for i in range(num_repeats) if i in avg]),
                    "std_ate_pred_med_horizon_val":  np.std( [avg[i]["ate_pred_med_horizon_val"] for i in range(num_repeats) if i in avg]),
                    "mean_ate_bias_med_horizon_val": np.mean([avg[i]["ate_med_horizon_bias_val"] for i in range(num_repeats) if i in avg]),
                    "std_ate_bias_med_horizon_val":  np.std( [avg[i]["ate_med_horizon_bias_val"] for i in range(num_repeats) if i in avg]),
                    # test
                    "mean_cate_mse_med_horizon": np.mean([avg[i]["cate_mse_med_horizon"] for i in range(num_repeats) if i in avg]),
                    "std_cate_mse_med_horizon":  np.std( [avg[i]["cate_mse_med_horizon"] for i in range(num_repeats) if i in avg]),
                    "mean_ate_pred_med_horizon": np.mean([avg[i]["ate_pred_med_horizon"] for i in range(num_repeats) if i in avg]),
                    "std_ate_pred_med_horizon":  np.std( [avg[i]["ate_pred_med_horizon"] for i in range(num_repeats) if i in avg]),
                    "mean_ate_bias_med_horizon": np.mean([avg[i]["ate_bias_med_horizon"] for i in range(num_repeats) if i in avg]),
                    "std_ate_bias_med_horizon":  np.std( [avg[i]["ate_bias_med_horizon"] for i in range(num_repeats) if i in avg]),
                    # true ATE at med horizon
                    "mean_ate_true_med_horizon": np.mean([avg[i]["ate_true_med_horizon"] for i in range(num_repeats) if i in avg]),
                    "std_ate_true_med_horizon":  np.std( [avg[i]["ate_true_med_horizon"] for i in range(num_repeats) if i in avg]),
                })
            
            mean_cate_surv_mse_val = None
            std_cate_surv_mse_val = None
            mean_cate_surv_mse = None
            std_cate_surv_mse = None
            surv_horizons_avg = None

            if args.include_surv_probs:
                # collect per-repeat MSE vectors where available
                mse_surv_val_list = [
                    avg[i]["cate_surv_mse_val"]
                    for i in range(num_repeats)
                    if i in avg and ("cate_surv_mse_val" in avg[i]) and (avg[i]["cate_surv_mse_val"] is not None)
                ]
                mse_surv_test_list = [
                    avg[i]["cate_surv_mse"]
                    for i in range(num_repeats)
                    if i in avg and ("cate_surv_mse" in avg[i]) and (avg[i]["cate_surv_mse"] is not None)
                ]
                horizons_list = [
                    avg[i]["surv_horizons"]
                    for i in range(num_repeats)
                    if i in avg and ("surv_horizons" in avg[i])
                ]

                if len(mse_surv_val_list) > 0:
                    mean_cate_surv_mse_val = np.nanmean(mse_surv_val_list, axis=0)
                    std_cate_surv_mse_val = np.nanstd(mse_surv_val_list, axis=0)
                if len(mse_surv_test_list) > 0:
                    mean_cate_surv_mse = np.nanmean(mse_surv_test_list, axis=0)
                    std_cate_surv_mse = np.nanstd(mse_surv_test_list, axis=0)
                if len(horizons_list) > 0:
                    surv_horizons_avg = horizons_list[0]
            
            if args.include_surv_probs and (mean_cate_surv_mse is not None):
                avg_dict.update({
                    "surv_horizons": surv_horizons_avg,           # (3,)
                    "mean_cate_surv_mse_val": mean_cate_surv_mse_val,
                    "std_cate_surv_mse_val": std_cate_surv_mse_val,
                    "mean_cate_surv_mse": mean_cate_surv_mse,
                    "std_cate_surv_mse": std_cate_surv_mse,
                })
            
            results_dict[config_name][scenario_key]["average"] = avg_dict
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
    parser.add_argument("--survival_metric", type=str, default="mean", choices=["median", "mean"]) # TODO: change the arg name to be `target`?
    parser.add_argument("--max_time", type=float, default=None, help="max time horizon for RMST calculation. None means using the maximum observed time.")
    parser.add_argument("--include_surv_probs", action="store_true", help="If set, include survival probabilities as estimands.")
    parser.add_argument(
        "--include_med_rmst",
        action="store_true",
        help="If set, include RMST with max horizon being median event time as an additional estimand."
    )
    parser.add_argument("--meta_learner", type=str, default="t_learner_survival", 
                        choices=["t_learner_survival", "s_learner_survival", "matching_learner_survival"])
    parser.add_argument("--base_survival_model", type=str, default="RandomSurvivalForest",
                        choices=["RandomSurvivalForest", "DeepSurv", "DeepHit"])
    parser.add_argument("--num_matches", type=int, default=5, help="Number of matches for matching learner")
    parser.add_argument("--save_model", action="store_true", 
                        help="If set, save the trained model. Default is False.")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    args = parser.parse_args()
    main(args)