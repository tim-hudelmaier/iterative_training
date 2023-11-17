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
    all_ids_str = "".join(sorted(id_list.copy())) if sorted else "".join(
        sample_ids.copy())
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

    df = pd.DataFrame()
    spectrum_ids = df["spectrum_id"].unique()
    for sample_ids in spectrum_ids.sample(frac=1 / n_samples):
        # md5 hash all sample spectrum ids
        # the idea here is to always hash all sorted ids, so I can check if a model with the
        # same spectrum ids has already been trained
        used_spectrum_ids_md5 = get_idx_md5(sample_ids, sort_ids=True)

        # save as pkl
        sample_df = df[df["spectrum_id"].isin(sample_ids)].copy()
        sample_df.to_pickle(
            sample_df_path / f"sample_{used_spectrum_ids_md5}.{file_extension}")

    # get all sample files
    sample_files = sample_df_path.glob("*.pkl")

    # get all possible ways to train
    train_combinations = permutations(sample_files, n_samples)

    # training
    for train_combination in train_combinations:
        pretrained_model_path = None
        model_output_path = None

        for i, sample_file in enumerate(train_combination):
            input_df = pd.concat([pd.read_pickle(f) for f in train_combination])

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
            output_path = ""
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
            other_output_path = ""
            train_run(other_config, other_output_path, df)
