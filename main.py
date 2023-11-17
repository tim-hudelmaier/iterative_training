"""Things to do:

- load input df with everything
- split input df into n samples
    - group by spectrum id -> avoid having decoy and targets in different training runs
    - group by & return random groups

- base run (n=1)
    - train model with 1 batch (base)
    - eval all batches (!)
- n-1 times
    - finetune with another batch
    - eval all batches (!)

- compare train with all PSMs
- different splits (size of spectra per split) go for n(splits) = [1:10]
- train with xgboost extend, random forest xtend and xgboost prune
"""
from itertools import permutations
import hashlib
from pathlib import Path

import pandas as pd

from pf_wrapper import train_run


def get_idx_md5(id_list: list, sort_ids=True):
    """Get md5 hash of all spectrum ids.

    Args:
        sorted (bool): whether to sort spectrum ids before hashing

    Returns:
        md5 hash of all spectrum ids
    """
    all_ids_str = "".join(sorted(id_list.copy())) if sort_ids else "".join(
        id_list.copy())
    return hashlib.md5(all_ids_str.encode()).hexdigest()


if __name__ == "__main__":
    n_samples = 5
    model_type = "xgboost"
    additional_estimators = 50
    file_extension = "json" if model_type == "xgboost" else "pkl"
    universal_feature_cols = False

    base_path = Path("./")
    sample_df_path = base_path / "sample_dfs"

    if not sample_df_path.exists():
        sample_df_path.mkdir()

    models_dir = base_path / "models"
    if not models_dir.exists():
        models_dir.mkdir()

    results_dir = base_path / "results"
    if not results_dir.exists():
        results_dir.mkdir()

    evals_dir = base_path / "evals"
    if not evals_dir.exists():
        evals_dir.mkdir()

    df = pd.DataFrame()
    spectrum_ids = df["spectrum_id"].unique()

    all_spectrum_ids_md5 = get_idx_md5(spectrum_ids, sort_ids=True)

    for sample_ids in spectrum_ids.sample(frac=1 / n_samples):
        # md5 hash all sample spectrum ids
        used_spectrum_ids_md5 = get_idx_md5(sample_ids, sort_ids=True)

        # save as pkl
        sample_df = df[df["spectrum_id"].isin(sample_ids)].copy()
        sample_df.to_pickle(
            sample_df_path / f"sample_{used_spectrum_ids_md5}.{file_extension}")

    # get all sample files
    sample_files = [p for p in sample_df_path.glob("*.pkl")]

    # get all possible ways to train
    train_combinations = permutations(sample_files, n_samples)

    # training
    for train_combination in train_combinations:
        pretrained_model_path = None
        model_output_path = None

        train_path_md5 = get_idx_md5([str(f) for f in train_combination],
                                     sort_ids=False)

        for i, sample_file in enumerate(train_combination):
            if i == len(train_combination) - 1:
                break

            input_df = pd.read_pickle(sample_file)

            # calculate hashes
            input_spectrum_ids = input_df["spectrum_id"].unique()
            input_spectrum_ids_md5 = get_idx_md5(input_spectrum_ids, sort_ids=True)

            if pretrained_model_path is not None:
                model_md5_str = pretrained_model_path.split("_")[-1].split(".")[0]
                new_model_md5 = get_idx_md5([model_md5_str, input_spectrum_ids_md5],
                                            sort_ids=False, )
            else:
                new_model_md5 = input_spectrum_ids_md5

            # check if model has already been trained
            if (models_dir / f"model_{new_model_md5}.{file_extension}").exists():
                continue

            model_output_path = models_dir / f"model_{new_model_md5}.{file_extension}"

            # train model
            config = {
                "conf": {
                    "model_type": model_type,
                    "mode": "train" if i == 0 else "finetune",
                    "additional_estimators": 0 if i == 0 else additional_estimators,
                    "pretrained_model_path": pretrained_model_path,
                    "model_output_path": model_output_path,
                },
                "universal_feature_cols": universal_feature_cols,
            }
            output_path = results_dir / f"train__path_{train_path_md5}__iteration_{i}__data_{new_model_md5}.csv"
            train_run(config, output_path, input_df)

            # set pretrained model path for next run
            pretrained_model_path = model_output_path

            # eval using all data
            other_config = {
                "conf": {
                    "model_type": model_type,
                    "mode": "eval",
                    "pretrained_model_path": pretrained_model_path,
                },
                "universal_feature_cols": False,
            }
            eval_df = pd.read_pickle(train_combination[i + 1])
            eval_data_md5 = get_idx_md5(eval_df["spectrum_id"].unique(), sort_ids=True)
            eval_output_path = evals_dir / f"eval__path_{train_path_md5}__iteration_{i}__data_{eval_data_md5}.csv"
            train_run(other_config, eval_output_path, eval_df)

# get full eval df by concatenating all eval dfs for one iteration
for iteration in range(n_samples):
    eval_dfs = [p for p in evals_dir.glob(f"*iteration_{iteration}*")]
    eval_md5s = [p.name.split("__")[-1].split(".")[0] for p in eval_dfs]

    eval_df = pd.concat([pd.read_csv(p) for p in eval_dfs])

    # check all spectrums have been evaluated
    eval_spectrum_ids = eval_df["spectrum_id"].unique()
    eval_spectrum_ids_md5 = get_idx_md5(eval_spectrum_ids, sort_ids=True)

    if eval_spectrum_ids_md5 != all_spectrum_ids_md5:
        raise ValueError("Not all spectrums have been evaluated!")

    # drop rank, top_target, q-value columns from eval runs
    rank_cols = [c for c in eval_df.columns if c.startswith("rank_")]
    top_target_cols = [c for c in eval_df.columns if c.startswith("top_target_")]
    q_value_cols = [c for c in eval_df.columns if c.startswith("q-value_")]
    eval_df.drop(rank_cols + top_target_cols + q_value_cols, axis=1, inplace=True)

    consolidate_evals_md5 = get_idx_md5(eval_md5s, sort_ids=True)
    eval_df.to_csv(
        evals_dir / f"consolidated_eval__data_{consolidate_evals_md5}__iteration_{iteration}.csv"
    )
