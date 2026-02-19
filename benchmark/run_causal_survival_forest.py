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
        else "semi-synthetic" if dataset_name in ["mimic_syn", "mimic_syn_2", "actg_syn"]
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
        failure_times_grid = None
    elif dataset_name == 'actg_syn':
        failure_times_grid_size = 200
        horizon, min_node_size = 30, 18
        failure_times_grid = None
    elif dataset_name == 'mimic_syn':
        failure_times_grid_size = 200
        horizon, min_node_size = 40, 18
        failure_times_grid = None
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
        failure_times_grid = None


    experiment_setups, experiment_repeat_setups = load_data(dataset_name=dataset_name, data_dir=data_dir)
    
    output_pickle_path = os.path.join(result_dir, dataset_type, f'models_causal_survival/causal_survival_forest/')
    output_pickle_path += f"{dataset_name}_causal_survival_forest_repeats_{args.num_repeats}_train_{train_size_str}.pkl"
    if args.exp_name != "":
        output_pickle_path = output_pickle_path.replace(".pkl", f"_{args.exp_name}.pkl")
    os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
    print("Output results path:", output_pickle_path)


    if os.path.exists(output_pickle_path):
        print("Loading results from existing file.")
        with open(output_pickle_path, 'rb') as f:
            results_dict = pickle.load(f)
    else:
        print("Results file not found, creating new file.")
        results_dict = {}

    # ----------- small helper for safe averaging (NEW) -----------
    def _safe_agg(avg_dict, key, num_repeats, stat="mean"):
        """Compute mean/std over repeats only if key exists."""
        vals = [
            avg_dict[i][key]
            for i in range(num_repeats)
            if i in avg_dict and key in avg_dict[i]
        ]
        if len(vals) == 0:
            return np.nan
        return np.mean(vals) if stat == "mean" else np.std(vals)

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
                test_size=test_size,
                include_surv_probs=args.include_surv_probs,
                include_rmst_med_horizon=args.include_med_rmst
            )
            if scenario_key not in results_dict[config_name]:
                results_dict[config_name][scenario_key] = {}

            ate_true = dataset_summary['ate'] # RMST @ max horizon
            ate_true_rmst_max = dataset_summary['ate']
            event_time_25pct = dataset_summary['event_time_25pct']
            event_time_50pct = dataset_summary['event_time_median']
            event_time_75pct = dataset_summary['event_time_75pct']
            ate_true_rmst_med = dataset_summary.get('ate_med_horizon', None)
            ate_true_surv_t25 = dataset_summary.get('ate_p_surv_t25', None)
            ate_true_surv_t50 = dataset_summary.get('ate_p_surv_t50', None)
            ate_true_surv_t75 = dataset_summary.get('ate_p_surv_t75', None)


            for rand_idx in range(num_repeats):
                # X_train, W_train, Y_train, cate_true_train = split_dict[rand_idx]['train']
                # X_val, W_val, Y_val, cate_true_val = split_dict[rand_idx]['val']
                # X_test, W_test, Y_test, cate_true_test = split_dict[rand_idx]['test']
                train_tuple = split_dict[rand_idx]['train']
                val_tuple   = split_dict[rand_idx]['val']
                test_tuple  = split_dict[rand_idx]['test']

                def _unpack(tuple_):
                    """
                    Handle variable outputs from prepare_data_split:
                    - (X, W, Y, cate)
                    - (X, W, Y, cate, cate_med)
                    - (X, W, Y, cate, cate_surv)
                    - (X, W, Y, cate, cate_med, cate_surv)
                    """
                    if len(tuple_) == 4:
                        X, W, Y, cate = tuple_
                        cate_med = None
                        cate_surv = None
                    elif len(tuple_) == 5:
                        X, W, Y, cate, extra = tuple_
                        if extra.ndim == 1:
                            cate_med = extra
                            cate_surv = None
                        else:
                            cate_med = None
                            cate_surv = extra
                    elif len(tuple_) == 6:
                        X, W, Y, cate, cate_med, cate_surv = tuple_
                    else:
                        raise ValueError(f"Unexpected split tuple length: {len(tuple_)}")
                    return X, W, Y, cate, cate_med, cate_surv

                (X_train, W_train, Y_train,
                cate_true_train,
                cate_true_med_train,
                cate_true_surv_train) = _unpack(split_dict[rand_idx]['train'])

                (X_val, W_val, Y_val,
                cate_true_val,
                cate_true_med_val,
                cate_true_surv_val) = _unpack(split_dict[rand_idx]['val'])

                (X_test, W_test, Y_test,
                cate_true_test,
                cate_true_med_test,
                cate_true_surv_test) = _unpack(split_dict[rand_idx]['test'])

                val_size_ = Y_val.shape[0]
                Y_val_test = np.vstack((Y_val, Y_test))
                has_med = cate_true_med_train is not None
                has_surv = cate_true_surv_train is not None

                max_time = Y_train[:, 0].max()
                if horizon is None:
                    horizon_rmst_max = float(max_time)
                else:
                    horizon_rmst_max = float(horizon)
                

                if rand_idx in results_dict[config_name][scenario_key]:
                    # t_ = time.time()
                    # learner.load_model(checkpoint_path)
                    # print(f'Took {(time.time() - t_):.0f} seconds to load the model')
                    runtime = results_dict[config_name][scenario_key][rand_idx]["runtime"]
                    print(f'\ttraining time from previous run: {runtime:.0f} seconds')
                    continue
                
                # ------------------ FIT all required forests ------------------
                train_start = time.time()

                # (1) RMST at max horizomn
                if dataset_name == 'synthetic':
                    csf_rmst_max = CausalSurvivalForestGRF(
                        failure_times_grid_size=failure_times_grid_size,
                        horizon=None,
                        target="RMST",
                        seed=2025 + rand_idx
                    )
                else:
                    csf_rmst_max = CausalSurvivalForestGRF(
                        failure_times_grid_size=failure_times_grid_size,
                        horizon=horizon_rmst_max,
                        target="RMST",
                        min_node_size=min_node_size,
                        seed=2025 + rand_idx
                    )

                if failure_times_grid is not None and dataset_name in ['twin30', 'twin180']:
                    csf_rmst_max.fit(X_train, W_train, Y_train, failure_times_grid=failure_times_grid)
                else:
                    csf_rmst_max.fit(X_train, W_train, Y_train)


                # (2) RMST at median horizon (optional)
                csf_rmst_med = None
                horizon_rmst_med = None
                if args.include_med_rmst and has_med and (event_time_50pct is not None):
                    horizon_rmst_med = float(event_time_50pct)
                    csf_rmst_med = CausalSurvivalForestGRF(
                        failure_times_grid_size=failure_times_grid_size,
                        horizon=horizon_rmst_med,
                        target="RMST",
                        min_node_size=min_node_size,
                        seed=2025 + rand_idx
                    )
                    if failure_times_grid is not None and dataset_name in ['twin30', 'twin180']:
                        csf_rmst_med.fit(X_train, W_train, Y_train, failure_times_grid=failure_times_grid)
                    else:
                        csf_rmst_med.fit(X_train, W_train, Y_train)
                elif args.include_med_rmst and not has_med:
                    print(f"[Warning][{config_name}][{scenario_key}] "
                          f"include_med_rmst=True but no cate_true_med_horizon in data; skipping RMST-median CSF.")
                    

                # (3) Survival probability models (optional)
                csf_surv = {}  # keys: 't25','t50','t75'
                horizons_surv = {}
                if args.include_surv_probs and has_surv:
                    # horizon times from summary
                    if event_time_25pct is not None:
                        horizons_surv['t25'] = float(event_time_25pct)
                    if event_time_50pct is not None:
                        horizons_surv['t50'] = float(event_time_50pct)
                    if event_time_75pct is not None:
                        horizons_surv['t75'] = float(event_time_75pct)

                    for label in ['t25', 't50', 't75']:
                        if label not in horizons_surv:
                            continue
                        h_val = horizons_surv[label]
                        csf_surv[label] = CausalSurvivalForestGRF(
                            failure_times_grid_size=failure_times_grid_size,
                            horizon=h_val,
                            target="survival.probability",
                            min_node_size=min_node_size,
                            seed=2025 + rand_idx
                        )
                        if failure_times_grid is not None and dataset_name in ['twin30', 'twin180']:
                            csf_surv[label].fit(X_train, W_train, Y_train, failure_times_grid=failure_times_grid)
                        else:
                            csf_surv[label].fit(X_train, W_train, Y_train)
                elif args.include_surv_probs and not has_surv:
                    print(f"[Warning][{config_name}][{scenario_key}] "
                          f"include_surv_probs=True but no survival-prob labels; skipping CSF survival targets.")

                train_end = time.time()
                runtime = train_end - train_start

                # ------------------- EVALUATE all models -------------------
                infer_start = time.time()

                # (1) RMST-max
                mse_val, cate_val_pred, ate_val_pred = csf_rmst_max.evaluate(X_val, cate_true_val, W_val)
                mse_test, cate_test_pred, ate_test_pred = csf_rmst_max.evaluate(X_test, cate_true_test, W_test)

                result_entry = {
                    "ate_true": ate_true_rmst_max,
                    "runtime": runtime,
                    # base: val
                    "cate_true_val": cate_true_val,
                    "cate_pred_val": cate_val_pred,
                    "ate_pred_val": ate_val_pred,
                    "cate_mse_val": mse_val,
                    "ate_bias_val": ate_val_pred - ate_true_rmst_max,
                    "ate_statistics_val": ate_val_pred,
                    # base: test
                    "cate_true": cate_true_test,
                    "cate_pred": cate_test_pred,
                    "ate_pred": ate_test_pred,
                    "cate_mse": mse_test,
                    "ate_bias": ate_test_pred - ate_true_rmst_max,
                    "ate_statistics": ate_test_pred,
                }

                # (2) RMST-median evaluation (if fitted)
                if csf_rmst_med is not None and cate_true_med_val is not None:
                    mse_val_med, cate_val_pred_med, ate_val_pred_med = csf_rmst_med.evaluate(
                        X_val, cate_true_med_val, W_val
                    )
                    mse_test_med, cate_test_pred_med, ate_test_pred_med = csf_rmst_med.evaluate(
                        X_test, cate_true_med_test, W_test
                    )

                    # use dataset_summary ate_med_horizon if present
                    ate_true_med = ate_true_rmst_med if ate_true_rmst_med is not None else np.mean(cate_true_med_test)

                    result_entry.update({
                        "ate_true_rmst_med": ate_true_med,
                        # val
                        "cate_true_val_rmst_med": cate_true_med_val,
                        "cate_pred_val_rmst_med": cate_val_pred_med,
                        "cate_mse_val_rmst_med": mse_val_med,
                        "ate_pred_val_rmst_med": ate_val_pred_med,
                        "ate_bias_val_rmst_med": ate_val_pred_med - ate_true_med,
                        # test
                        "cate_true_rmst_med": cate_true_med_test,
                        "cate_pred_rmst_med": cate_test_pred_med,
                        "cate_mse_rmst_med": mse_test_med,
                        "ate_pred_rmst_med": ate_test_pred_med,
                        "ate_bias_rmst_med": ate_test_pred_med - ate_true_med,
                    })

                # (3) Survival-probability evaluation
                if csf_surv and cate_true_surv_val is not None:
                    # cate_true_surv_*: (n, 3), columns [t25, t50, t75]
                    label_to_idx = {'t25': 0, 't50': 1, 't75': 2}

                    n_val = cate_true_surv_val.shape[0]
                    n_test = cate_true_surv_test.shape[0]

                    # Initialize prediction matrices (n, 3)
                    cate_surv_val_pred = np.zeros_like(cate_true_surv_val, dtype=float)
                    cate_surv_test_pred = np.zeros_like(cate_true_surv_test, dtype=float)

                    # For each horizon, fill in the corresponding column with CSF predictions
                    for label, j in label_to_idx.items():
                        if label not in csf_surv:
                            # leave column as zeros if model wasn't fitted (shouldn't happen
                            # if data is consistent, but safer)
                            continue

                        cate_val_true_h = cate_true_surv_val[:, j]
                        cate_test_true_h = cate_true_surv_test[:, j]

                        mse_val_h, cate_val_pred_h, ate_val_pred_h = csf_surv[label].evaluate(
                            X_val, cate_val_true_h, W_val
                        )
                        mse_test_h, cate_test_pred_h, ate_test_pred_h = csf_surv[label].evaluate(
                            X_test, cate_test_true_h, W_test
                        )

                        # Fill predictions
                        cate_surv_val_pred[:, j] = cate_val_pred_h
                        cate_surv_test_pred[:, j] = cate_test_pred_h

                    # Now compute MSE per horizon, matching meta-learner logic
                    mse_surv_val = np.mean(
                        (cate_true_surv_val - cate_surv_val_pred) ** 2,
                        axis=0
                    )
                    mse_surv_test = np.mean(
                        (cate_true_surv_test - cate_surv_test_pred) ** 2,
                        axis=0
                    )

                    # Store in the same key structure as meta-learner
                    result_entry.update({
                        "surv_horizons": np.array(
                            [event_time_25pct, event_time_50pct, event_time_75pct],
                            dtype=float,
                        ),
                        "cate_surv_true_val": cate_true_surv_val,
                        "cate_surv_true": cate_true_surv_test,
                        "cate_surv_pred_val": cate_surv_val_pred,
                        "cate_surv_pred": cate_surv_test_pred,
                        "cate_surv_mse_val": mse_surv_val,
                        "cate_surv_mse": mse_surv_test,
                    })

                infer_end = time.time()
                inference_time = infer_end - infer_start
                result_entry["inference_time"] = inference_time

                results_dict[config_name][scenario_key][rand_idx] = result_entry

                print(f'\ttraining time: {runtime:.0f} seconds; inference time: {inference_time:.0f} seconds')

                with open(output_pickle_path, "wb") as f:
                    pickle.dump(results_dict, f)

                # print(f"Completed {config_name}, {scenario_key}, repeat {rand_idx}: CATE MSE={mse_test:.4f}, ATE True={ate_true:.4f}, ATE Pred={ate_test_pred.mean_point:.4f}")
            

            if len(results_dict[config_name][scenario_key]) == 0:
                print(f"[Warning]: No valid results for {config_name}, {scenario_key}. Skipping.")
                continue

            # Save results to the setup dictionary
            avg = results_dict[config_name][scenario_key]
            average_entry = {
                # base RMST @ max horizon
                "mean_cate_mse_val": _safe_agg(avg, "cate_mse_val", num_repeats, "mean"),
                "std_cate_mse_val":  _safe_agg(avg, "cate_mse_val", num_repeats, "std"),
                "mean_ate_pred_val": _safe_agg(avg, "ate_pred_val", num_repeats, "mean"),
                "std_ate_pred_val":  _safe_agg(avg, "ate_pred_val", num_repeats, "std"),
                "mean_ate_bias_val": _safe_agg(avg, "ate_bias_val", num_repeats, "mean"),
                "std_ate_bias_val":  _safe_agg(avg, "ate_bias_val", num_repeats, "std"),

                "mean_cate_mse": _safe_agg(avg, "cate_mse", num_repeats, "mean"),
                "std_cate_mse":  _safe_agg(avg, "cate_mse", num_repeats, "std"),
                "mean_ate_pred": _safe_agg(avg, "ate_pred", num_repeats, "mean"),
                "std_ate_pred":  _safe_agg(avg, "ate_pred", num_repeats, "std"),
                "mean_ate_bias": _safe_agg(avg, "ate_bias", num_repeats, "mean"),
                "std_ate_bias":  _safe_agg(avg, "ate_bias", num_repeats, "std"),

                "mean_ate_true": _safe_agg(avg, "ate_true", num_repeats, "mean"),
                "std_ate_true":  _safe_agg(avg, "ate_true", num_repeats, "std"),
                "runtime":       _safe_agg(avg, "runtime", num_repeats, "mean"),
            }


            # >>> aggregate RMST-median if present
            if any("cate_mse_val_rmst_med" in avg[i] for i in avg if isinstance(i, int)):
                average_entry.update({
                    "mean_cate_mse_val_rmst_med": _safe_agg(avg, "cate_mse_val_rmst_med", num_repeats, "mean"),
                    "std_cate_mse_val_rmst_med":  _safe_agg(avg, "cate_mse_val_rmst_med", num_repeats, "std"),
                    "mean_ate_pred_val_rmst_med": _safe_agg(avg, "ate_pred_val_rmst_med", num_repeats, "mean"),
                    "std_ate_pred_val_rmst_med":  _safe_agg(avg, "ate_pred_val_rmst_med", num_repeats, "std"),
                    "mean_ate_bias_val_rmst_med": _safe_agg(avg, "ate_bias_val_rmst_med", num_repeats, "mean"),
                    "std_ate_bias_val_rmst_med":  _safe_agg(avg, "ate_bias_val_rmst_med", num_repeats, "std"),

                    "mean_cate_mse_rmst_med": _safe_agg(avg, "cate_mse_rmst_med", num_repeats, "mean"),
                    "std_cate_mse_rmst_med":  _safe_agg(avg, "cate_mse_rmst_med", num_repeats, "std"),
                    "mean_ate_pred_rmst_med": _safe_agg(avg, "ate_pred_rmst_med", num_repeats, "mean"),
                    "std_ate_pred_rmst_med":  _safe_agg(avg, "ate_pred_rmst_med", num_repeats, "std"),
                    "mean_ate_bias_rmst_med": _safe_agg(avg, "ate_bias_rmst_med", num_repeats, "mean"),
                    "std_ate_bias_rmst_med":  _safe_agg(avg, "ate_bias_rmst_med", num_repeats, "std"),
                })


            mean_cate_surv_mse_val = None
            std_cate_surv_mse_val = None
            mean_cate_surv_mse = None
            std_cate_surv_mse = None
            surv_horizons_avg = None

            # >>> aggregate survival-probability metrics if present
            if args.include_surv_probs:
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
                average_entry.update({
                    "surv_horizons": surv_horizons_avg,
                    "mean_cate_surv_mse_val": mean_cate_surv_mse_val,
                    "std_cate_surv_mse_val": std_cate_surv_mse_val,
                    "mean_cate_surv_mse": mean_cate_surv_mse,
                    "std_cate_surv_mse": std_cate_surv_mse,
                })

            results_dict[config_name][scenario_key]["average"] = average_entry

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
    parser.add_argument(
        "--include_surv_probs",
        action="store_true",
        help="If set, include survival probabilities (t25/t50/t75) as additional estimands."
    )
    parser.add_argument(
        "--include_med_rmst",
        action="store_true",
        help="If set, include RMST with max horizon being median event time as an additional estimand."
    )
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    # We do not provide model saving/loading for CSF as it is from an R module
    args = parser.parse_args()
    main(args)