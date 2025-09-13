import argparse
import os
import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
from models_causal_survival_meta.meta_learners_survival import TLearnerSurvival, SLearnerSurvival, MatchingLearnerSurvival
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

def main(args):
    """Main function to run the survival meta-learner experiments."""
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
        for scenario in range(1, 2):
            result = pd.read_csv(path)
            if result is not None:
                scenario_dict[f"scenario_{scenario}"] = result
        experiment_setups[base_name] = scenario_dict

    output_pickle_path = f"results/real_data/models_causal_survival_meta/{args.meta_learner}/"
    output_pickle_path += f"{args.meta_learner}_{args.base_survival_model}_repeats_{args.num_repeats}.pkl"
    print("Output results path:", output_pickle_path)

    # Define base survival models to use
    base_model = args.base_survival_model
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

    for setup_name, setup_dict in tqdm(experiment_setups.items(), desc="Experiment Setups"):
        results_dict[setup_name] = {}
        hiv_dataset_idx = int(setup_name[-1])
        experiment_repeat_setup = experiment_repeat_setups[hiv_dataset_idx-1]
        for scenario_key in tqdm(setup_dict, desc=f"{setup_name} Scenarios"):
            dataset_df = setup_dict[scenario_key]
            split_dict = prepare_actg_data_split(dataset_df, X_cols, W_col, cate_base_col, experiment_repeat_setup)
            results_dict[setup_name][scenario_key] = {}

            start_time = time.time()

            for rand_idx in range(NUM_REPEATS_TO_INCLUDE):
                X_train, W_train, Y_train, X_test, W_test, Y_test, cate_test_true = split_dict[rand_idx]

                max_time = Y_train[:, 0].max()
                
                # Initialize the appropriate meta-learner
                if args.meta_learner == "t_learner_survival":
                    learner = TLearnerSurvival(
                        base_model_name=base_model,
                        base_model_grid=hyperparameter_grids,
                        metric=args.survival_metric,
                        max_time=max_time
                    )
                elif args.meta_learner == "s_learner_survival":
                    learner = SLearnerSurvival(
                        base_model_name=base_model,
                        base_model_grid=hyperparameter_grids,
                        metric=args.survival_metric,
                        max_time=max_time
                    )
                elif args.meta_learner == "matching_learner_survival":
                    learner = MatchingLearnerSurvival(
                        base_model_name=base_model,
                        base_model_grid=hyperparameter_grids,
                        metric=args.survival_metric,
                        num_matches=args.num_matches,
                        max_time=max_time
                    )

                if args.meta_learner == "t_learner_survival":
                    if Y_train[W_train == 1, 1].sum() <= 1:
                        print(f"[Warning]: For {args.meta_learner}, No event in treatment group. Skipping iteration {rand_idx}.")
                        continue
                    if Y_train[W_train == 0, 1].sum() <= 1:
                        print(f"[Warning]: For {args.meta_learner}, No event in control group. Skipping iteration {rand_idx}.")
                        continue

                # Fit the learner
                learner.fit(X_train, W_train, Y_train)

                ate_true = TRUE_ATE.get((setup_name, scenario_key), cate_test_true.mean())
                
                # Evaluate base survival models on test data
                base_model_eval = learner.evaluate_test(X_test, Y_test, W_test)
                
                # Evaluate causal effect predictions
                mse_test, cate_test_pred, ate_test_pred = learner.evaluate(X_test, cate_test_true, W_test)

                results_dict[setup_name][scenario_key][rand_idx] = {
                    "cate_true": cate_test_true,
                    "cate_pred": cate_test_pred,
                    "ate_true": ate_true,
                    "ate_pred": ate_test_pred,
                    "cate_mse": mse_test,
                    "ate_bias": ate_test_pred - ate_true,
                    "base_model_eval": base_model_eval  # Store base model evaluation results
                }

            end_time = time.time()
            avg = results_dict[setup_name][scenario_key]
            if len(avg) == 0:
                base_model_eval_performance = {}
            else:
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
                
            results_dict[setup_name][scenario_key]["average"] = {
                "mean_cate_mse": np.mean([avg[i]["cate_mse"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "std_cate_mse": np.std([avg[i]["cate_mse"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "mean_ate_pred": np.mean([avg[i]["ate_pred"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "std_ate_pred": np.std([avg[i]["ate_pred"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "mean_ate_true": np.mean([avg[i]["ate_true"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "std_ate_true": np.std([avg[i]["ate_true"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "mean_ate_bias": np.mean([avg[i]["ate_bias"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "std_ate_bias": np.std([avg[i]["ate_bias"] for i in range(NUM_REPEATS_TO_INCLUDE) if i in avg]),
                "runtime": (end_time - start_time) / len(avg) if len(avg) > 0 else 0,
                "base_model_eval" : base_model_eval_performance
                }

            with open(output_pickle_path, "wb") as f:
                pickle.dump(results_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--train_size", type=float, default=0.75)
    parser.add_argument("--survival_metric", type=str, default="mean", choices=["median", "mean"])
    parser.add_argument("--meta_learner", type=str, default="t_learner_survival", 
                        choices=["t_learner_survival", "s_learner_survival", "matching_learner_survival"])
    parser.add_argument("--base_survival_model", type=str, default="RandomSurvivalForest",
                        choices=["RandomSurvivalForest", "DeepSurv", "DeepHit"])
    parser.add_argument("--num_matches", type=int, default=5, help="Number of matches for matching learner")
    args = parser.parse_args()
    main(args)