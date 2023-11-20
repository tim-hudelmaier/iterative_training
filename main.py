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

from pathlib import Path

import pandas as pd

from peptide_forest import PeptideForest
from iter_helpers.pf_wrapper import train_run, consolidate_evals
from iter_helpers.iter_helpers import (
    get_idx_md5,
    create_dirs,
    drop_cols,
    generate_eval_df,
    generate_and_pickle_samples,
    generate_next_train_run,
)

if __name__ == "__main__":
    n_samples = 5
    model_type = "xgboost"
    additional_estimators = 50
    file_extension = "json" if model_type == "xgboost" else "pkl"
    universal_feature_cols = False

    base_dir = Path("./")
    dir_dict = create_dirs(base_dir)

    base_pf = PeptideForest(
        config_path="config__PXD021874_total.json",
        output=None
    )
    base_pf.prep_ursgal_csvs()
    base_pf.calc_features()

    df = base_pf.input_df.copy()

    all_spectrum_ids_md5 = generate_and_pickle_samples(
        df=df,
        sample_group_col="spectrum_id",
        n_samples=n_samples,
        sample_dir=dir_dict["sample_dir"],
        file_extension=file_extension,
    )

    sample_files = [p for p in dir_dict["sample_dir"].glob("*.pkl")]

    pretrained_model_path = None
    model_output_path = None

    # training
    for i, train_sample_file, eval_sample_file, train_path_md5, finished_path in generate_next_train_run(
            sample_files):
        if finished_path:
            pretrained_model_path = None
            model_output_path = None
            continue

        input_df = pd.read_pickle(train_sample_file)

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
        if (dir_dict[
                "models_dir"] / f"model_{new_model_md5}.{file_extension}").exists():
            continue

        model_output_path = dir_dict[
                                "models_dir"] / f"model_{new_model_md5}.{file_extension}"

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
        output_path = dir_dict[
                          "results_dir"] / f"train__path_{train_path_md5}__iteration_{i}__data_{new_model_md5}.csv"
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
        eval_df = pd.read_pickle(eval_sample_file)
        eval_data_md5 = get_idx_md5(eval_df["spectrum_id"].unique(), sort_ids=True)
        eval_output_path = dir_dict[
                               "evals_dir"] / f"eval__path_{train_path_md5}__iteration_{i}__data_{eval_data_md5}.csv"
        train_run(other_config, eval_output_path, eval_df)

    # get full eval df by concatenating all eval dfs for one iteration
    for i, eval_df, eval_data_md5 in generate_eval_df(n_samples, dir_dict["evals_dir"]):
        if eval_data_md5 != all_spectrum_ids_md5:
            raise ValueError("Not all spectrums have been evaluated!")

        # drop rank, top_target, q-value columns from eval runs
        eval_df = drop_cols(eval_df, ["rank_", "top_target_", "q-value_"])

        output_path = dir_dict[
                          "evals_dir"] / f"consolidated_eval__data_{eval_data_md5}__iteration_{i}.csv"
        consolidate_evals(
            config={"initial_engine": "some_engine"},
            output_path=output_path,
            trained_df=eval_df,
        )
