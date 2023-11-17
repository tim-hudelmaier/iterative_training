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

n_samples = 5
model_type = "xgboost"
additional_estimators = 50

sample_df_path = Path("./samples")

df = pd.DataFrame()
spectrum_ids = df["spectrum_id"].unique()
for sample_ids in spectrum_ids.sample(frac=1/n_samples):
    # md5 hash all sample spectrum ids
    # the idea here is to always hash all sorted ids, so I can check if a model with the
    # same spectrum ids has already been trained
    all_ids_str = "".join(sorted(sample_ids.copy()))
    used_spectrum_ids_md5 = hashlib.md5(all_ids_str.encode()).hexdigest()

    # save as pkl
    sample_df = df[df["spectrum_id"].isin(sample_ids)].copy()
    sample_df.to_pickle(sample_df_path / f"sample_{used_spectrum_ids_md5}.pkl")

# get all sample files
sample_files = sample_df_path.glob("*.pkl")

# get all possible ways to train
train_combinations = permutations(sample_files, n_samples)

# training
for train_combination in train_combinations:

    for i, sample_file in enumerate(train_combination):
        input_df = pd.concat([pd.read_pickle(f) for f in train_combination])

        # train model
        config = {
            "conf": {
                "model_type": model_type,
                "mode": "train" if i == 0 else "finetune",
                "additional_estimators": 0 if i == 0 else additional_estimators,
                "pretrained_model_path": output_path,
            },
            "universal_feature_cols": False,
        }
        output_path = ""
        train_run(config, output_path, input_df)

        # eval using all data
        other_config = {
            "conf": {
                "model_type": model_type,
                "mode": "eval",
                "pretrained_model_path": output_path,
            },
            "universal_feature_cols": False,
        }
        other_output_path = ""
        train_run(other_config, other_output_path, df)

