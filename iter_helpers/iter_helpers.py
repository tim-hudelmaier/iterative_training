from itertools import permutations
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def get_idx_md5(id_list: list, sort_ids=True):
    """Get md5 hash of all spectrum ids.

    Args:
        sorted (bool): whether to sort spectrum ids before hashing

    Returns:
        md5 hash of all spectrum ids
    """
    if isinstance(id_list, str):
        return hashlib.md5(id_list.encode()).hexdigest()
    elif isinstance(id_list, pd.Series):
        id_list = id_list.unique().tolist()
    else:
        ValueError(f"id_list must be a list or pd.Series, you provided {type(id_list)}")
    all_ids_str = (
        "".join(sorted(id_list.copy())) if sort_ids else "".join(id_list.copy())
    )
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


def generate_eval_df(n_iterations: int, evals_dir: Path, col="spectrum_id"):
    for iteration in range(n_iterations):
        eval_dfs = [p for p in evals_dir.glob(f"*iteration_{iteration}*")]

        eval_df = pd.concat([pd.read_csv(p) for p in eval_dfs])

        eval_spectrum_ids = eval_df[col].unique()
        eval_md5 = get_idx_md5(eval_spectrum_ids, sort_ids=True)

        yield iteration, eval_df, eval_md5


def generate_and_pickle_samples(
    df: pd.DataFrame,
    sample_group_col: str,
    n_samples: int,
    sample_dir: Path,
    file_extension: str,
):
    """Generate and pickle samples from input df."""
    available_samples = pd.Series(df[sample_group_col].unique())

    all_samples_md5 = get_idx_md5(available_samples, sort_ids=True)

    shuffled_samples = available_samples.sample(frac=1).reset_index(drop=True)

    split_indices = np.array_split(shuffled_samples, n_samples)
    for samples in split_indices:
        # md5 hash all sample spectrum ids
        used_samples_md5 = get_idx_md5(samples, sort_ids=True)

        # save as pkl
        if isinstance(samples, str):
            samples = [samples]

        sample_df = df[df[sample_group_col].isin(samples)].copy()
        sample_df.to_pickle(sample_dir / f"sample_{used_samples_md5}.{file_extension}")

    return all_samples_md5


def generate_next_train_run(sample_files):
    train_combinations = permutations(sample_files, len(sample_files))

    for train_combination in train_combinations:
        train_path_md5 = get_idx_md5(
            [str(f) for f in train_combination], sort_ids=False
        )

        for i, train_sample_file in enumerate(train_combination):
            eval_sample_file = (
                train_combination[i + 1] if i < len(train_combination) - 1 else None
            )
            finished_path = True if i == len(train_combination) - 1 else False
            yield i, train_sample_file, eval_sample_file, train_path_md5, finished_path
