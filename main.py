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

from pf_wrapper import train_run, consolidate_evals


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


def create_dirs(base_dir: Path) -> dict:
    """Create necessary directories."""
    sample_dir = base_dir / "sample_dfs"
    if not sample_dir.exists():
        sample_dir.mkdir()

    models_dir = base_dir / "models"
    if not models_dir.exists():
        models_dir.mkdir()

    results_dir = base_dir / "results"
    if not results_dir.exists():
        results_dir.mkdir()

    evals_dir = base_dir / "evals"
    if not evals_dir.exists():
        evals_dir.mkdir()

    return {
        "sample_dir": sample_dir,
        "models_dir": models_dir,
        "results_dir": results_dir,
        "evals_dir": evals_dir,
    }


def drop_cols(df: pd.DataFrame, col_prefixes: list) -> pd.DataFrame:
    """Drop columns from dataframe."""
    cols_to_drop = []
    for col_prefix in col_prefixes:
        cols_to_drop += [c for c in df.columns if c.startswith(col_prefix)]
    df.drop(cols_to_drop, axis=1, inplace=True)
    return df


def generate_eval_df(n_iterations: int, evals_dir: Path):
    for iteration in range(n_iterations):
        eval_dfs = [p for p in evals_dir.glob(f"*iteration_{iteration}*")]

        eval_df = pd.concat([pd.read_csv(p) for p in eval_dfs])

        eval_spectrum_ids = eval_df["spectrum_id"].unique()
        eval_spectrum_ids_md5 = get_idx_md5(eval_spectrum_ids, sort_ids=True)

        yield iteration, eval_df, eval_spectrum_ids_md5


def generate_and_pickle_samples(
        df: pd.DataFrame,
        sample_group_col: str,
        n_samples: int,
        sample_dir: Path,
        file_extension: str
):
    """Generate and pickle samples from input df."""
    available_samples = df[sample_group_col].unique()

    all_samples_md5 = get_idx_md5(available_samples, sort_ids=True)

    for samples in available_samples.sample(frac=1 / n_samples):
        # md5 hash all sample spectrum ids
        used_samples_md5 = get_idx_md5(samples, sort_ids=True)

        # save as pkl
        sample_df = df[df[sample_group_col].isin(samples)].copy()
        sample_df.to_pickle(
            sample_dir / f"sample_{used_samples_md5}.{file_extension}"
        )

    return all_samples_md5


def generate_next_train_run(sample_files):
    train_combinations = permutations(sample_files, len(sample_files))

    for train_combination in train_combinations:
        train_path_md5 = get_idx_md5([str(f) for f in train_combination],
                                     sort_ids=False)

        for i, train_sample_file in enumerate(train_combination):
            eval_sample_file = train_combination[i + 1] if i < len(
                train_combination) - 1 else None
            finished_path = True if i == len(train_combination) - 1 else False
            yield i, train_sample_file, eval_sample_file, train_path_md5, finished_path


if __name__ == "__main__":
    n_samples = 5
    model_type = "xgboost"
    additional_estimators = 50
    file_extension = "json" if model_type == "xgboost" else "pkl"
    universal_feature_cols = False

    base_dir = Path("./")
    dir_dict = create_dirs(base_dir)

    df = pd.DataFrame()

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
        if (dir_dict["models_dir"] / f"model_{new_model_md5}.{file_extension}").exists():
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
