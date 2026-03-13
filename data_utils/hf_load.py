"""
Load SurvHTE-Bench from HuggingFace Hub.

Two interfaces:

1. load_data(dataset_name) → experiment_setups, experiment_repeat_setups
   (identical output to the original local load_data())

2. load_splits(dataset_name) → nested dict mirroring prepare_data_split output
   results[config_name][scenario_key][rand_idx]["train"|"val"|"test"]
       = (X, W, Y, cate_true)
"""

import json
import numpy as np
import pandas as pd
import datasets as hf_datasets

REPO_ID = "snoroozi/SurvHTE-Bench"

SCHEMA = {
    "synthetic": dict(
        X_cols=None,   # resolved dynamically: cols starting with X + digit
        W_col="W",
        y_cols=["observed_time", "event"],
        cate_col=None, # computed as T1 - T0
        idx_col="id",
    ),
    "actg_syn": dict(
        X_cols=["age","wtkg","hemo","homo","drugs","karnof","oprior","z30","zprior","preanti",
                "race","gender","str2","strat","symptom","treat","offtrt",
                "cd40","cd420","cd496","r","cd80","cd820"],
        W_col="W",
        y_cols=["observed_time", "event"],
        cate_col="true_cate",
        idx_col="idx",
    ),
    "twin": dict(
        X_cols=["anemia","cardiac","lung","diabetes","herpes","hydra","hemo","chyper","phyper",
                "eclamp","incervix","pre4000","preterm","renal","rh","uterine","othermr",
                "gestat","dmage","dmeduc","dmar","nprevist","adequacy",
                "dtotord","cigar","drink","wtgain",
                "pldel_2","pldel_3","pldel_4","pldel_5","resstatb_2","resstatb_3","resstatb_4",
                "mpcb_1","mpcb_2","mpcb_3","mpcb_4","mpcb_5","mpcb_6","mpcb_7","mpcb_8","mpcb_9"],
        W_col="W",
        y_cols=["observed_time", "event"],
        cate_col="true_cate",
        idx_col="idx",
    ),
    "actgHC": dict(
        X_cols=["gender","race","hemo","homo","drugs","str2","symptom","age","wtkg","karnof","cd40","cd80"],
        W_col="trt",
        y_cols=None,   # 10 versions: t0/e0 .. t9/e9 — caller slices Y[:, r*2:(r+1)*2]
        cate_col="cate_base",
        idx_col="id",
    ),
    "actgLC": dict(
        X_cols=["gender","race","hemo","homo","drugs","str2","symptom","age","wtkg","karnof","cd40","cd80"],
        W_col="trt",
        y_cols=["observed_time_month", "effect_non_censor"],
        cate_col="cate_base",
        idx_col="id",
    ),
}

# Mirrors SPLIT_SIZES in hf_upload.py
SPLIT_SIZES = {
    "synthetic": (5000, 2500, 2500),
    "actg_syn":  (0.50, 0.25, 0.25),
    "twin":      (0.50, 0.25, 0.25),
    "actgHC":    (0.50, 0.25, 0.25),
    "actgLC":    (0.50, 0.25, 0.25),
}


def load_data(
    dataset_name: str,
    repo_id: str = REPO_ID,
) -> tuple[dict, dict]:
    """Identical output to local load_data()."""

    setups_df = hf_datasets.load_dataset(repo_id, name=dataset_name, split="train").to_pandas()
    meta_cols = {"setup_key", "scenario", "summary_json", "metadata_json"}
    data_cols = [c for c in setups_df.columns if c not in meta_cols]

    experiment_setups: dict = {}
    for (setup_key, scenario), grp in setups_df.groupby(["setup_key", "scenario"], sort=False):
        info = {
            "dataset": grp[data_cols].reset_index(drop=True),
            "summary": json.loads(grp["summary_json"].iloc[0]),
        }
        if "metadata_json" in grp.columns and grp["metadata_json"].notna().any():
            info["metadata"] = json.loads(grp["metadata_json"].iloc[0])
        experiment_setups.setdefault(setup_key, {})[scenario] = info

    repeats_df = hf_datasets.load_dataset(repo_id, name=f"{dataset_name}_repeats", split="train").to_pandas()
    idx_cols = [c for c in repeats_df.columns if c != "repeat_key"]

    if dataset_name in ("actgHC", "actgLC"):
        experiment_repeat_setups: dict = {}
        for repeat_key, grp in repeats_df.groupby("repeat_key", sort=False):
            experiment_repeat_setups[repeat_key] = grp[idx_cols].reset_index(drop=True)
    else:
        experiment_repeat_setups = repeats_df[idx_cols].reset_index(drop=True)

    return experiment_setups, experiment_repeat_setups


def load_splits(
    dataset_name: str,
    num_repeats: int = 10,
    repo_id: str = REPO_ID,
) -> dict:
    """
    Returns nested dict mirroring the experiment loop:

        results[config_name][scenario_key][rand_idx]["train"|"val"|"test"]
            = (X, W, Y, cate_true)
    """
    schema   = SCHEMA[dataset_name]
    W_col    = schema["W_col"]
    y_cols   = schema["y_cols"]
    cate_col = schema["cate_col"]

    # Load all 30 splits once
    raw: dict[tuple, pd.DataFrame] = {}
    for split_type in ("train", "val", "test"):
        for r in range(num_repeats):
            raw[(split_type, r)] = hf_datasets.load_dataset(
                repo_id, name=f"{dataset_name}_splits", split=f"{split_type}_{r}"
            ).to_pandas()

    # Discover (config_name, scenario) pairs from train_0
    pairs = list(
        raw[("train", 0)].groupby(["setup_key", "scenario"], sort=False).groups.keys()
    )

    # Resolve X columns from train_0
    sample_df = raw[("train", 0)]
    if schema["X_cols"] is None:
        X_cols = [c for c in sample_df.columns if c.startswith("X") and c[1:].isdigit()]
    else:
        X_cols = schema["X_cols"]

    def _extract(df_grp: pd.DataFrame) -> tuple:
        X = df_grp[X_cols].to_numpy()
        W = df_grp[W_col].to_numpy()
        if y_cols is not None:
            Y = df_grp[y_cols].to_numpy()
        else:
            # actgHC: return all t/e cols, caller slices Y[:, r*2:(r+1)*2]
            te_cols = [col for i in range(10) for col in (f"t{i}", f"e{i}") if col in df_grp.columns]
            Y = df_grp[te_cols].to_numpy()
        if cate_col is not None and cate_col in df_grp.columns:
            cate_true = df_grp[cate_col].to_numpy()
        else:
            cate_true = (df_grp["T1"] - df_grp["T0"]).to_numpy()
        return (X, W, Y, cate_true)

    results: dict = {}
    for config_name, scenario_key in pairs:
        results.setdefault(config_name, {})[scenario_key] = {r: {} for r in range(num_repeats)}
        for split_type in ("train", "val", "test"):
            for r in range(num_repeats):
                df = raw[(split_type, r)]
                grp = df[
                    (df["setup_key"] == config_name) & (df["scenario"] == scenario_key)
                ].reset_index(drop=True)
                results[config_name][scenario_key][r][split_type] = _extract(grp)

    return results