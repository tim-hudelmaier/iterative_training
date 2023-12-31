from itertools import permutations
import hashlib
from pathlib import Path
import re

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
    elif isinstance(id_list, list):
        id_list = list(dict.fromkeys(id_list))
    elif isinstance(id_list, np.ndarray):
        id_list = np.unique(id_list).tolist()
    else:
        raise ValueError(
            f"id_list must be a list or pd.Series, you provided {type(id_list)}"
        )
    id_list = [str(i) for i in id_list]
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


def create_eval_df_sublists(eval_df_paths: list) -> list:
    """Create sublists of eval_df_paths based on data md5."""
    eval_df_sublists = []

    for i, eval_df_path in enumerate(eval_df_paths):
        if isinstance(eval_df_path, Path):
            match = re.search(r'__data_([A-Za-z0-9]+)', str(eval_df_path.name))
        elif isinstance(eval_df_path, str):
            match = re.search(r'__data_([A-Za-z0-9]+)', eval_df_path)
        else:
            raise TypeError(f"eval_df_paths elements have to be str or Path, you "
                            f"provided {type(eval_df_path)}")

        if match:
            data_md5 = match.group(1)
        else:
            continue

        if not eval_df_sublists:
            eval_df_sublists.append({data_md5: eval_df_path})
            continue

        path_inserted = False
        for i, sublist in enumerate(eval_df_sublists):
            if data_md5 not in list(sublist.keys()):
                eval_df_sublists[i][data_md5] = eval_df_path
                path_inserted = True
                break

        if not path_inserted:
            eval_df_sublists.append({data_md5: eval_df_path})

    # ensure only full length lists are returned
    max_length = max([len(list(sublist.keys())) for sublist in eval_df_sublists])

    return [list(sublist.values()) for sublist in eval_df_sublists if
            len(list(sublist.keys())) == max_length]


def concat_and_aggreagte_dfs(df_paths: list, idx_cols: list=None, score_cols: list=None) -> pd.DataFrame:
    if idx_cols is None:
        idx_cols = ["spectrum_id", "is_decoy"]
    if score_cols is None:
        score_cols = ["score"]

    df = pd.concat([pd.read_csv(p) for p in df_paths])

    # warn that cols not in idx_cols or score_cols are dropped
    for c in df.columns:
        if c not in idx_cols and c not in score_cols:
            print(f"Warning: dropping column {c} from eval_df.")

    return df.groupby(idx_cols)[score_cols].mean().reset_index()



def generate_eval_df(
        n_iterations: int, evals_dir: Path, method="consolidate",
        idx_cols=None, score_cols=None, col="spectrum_id",
):
    for iteration in range(n_iterations):
        # aggregate duplicates by idx_cols and average
        if method == "consolidate":
            eval_dfs = [p for p in evals_dir.glob(f"*iteration_{iteration}*")]

            eval_df = concat_and_aggreagte_dfs(
                df_paths=eval_dfs,
                idx_cols=idx_cols,
                score_cols=score_cols
            )

            eval_spectrum_ids = eval_df[col].unique()
            eval_md5 = get_idx_md5(eval_spectrum_ids, sort_ids=True)

            yield iteration, eval_df, eval_md5

        elif method == "rolling":
            eval_paths = [p for p in evals_dir.glob(f"*iteration_{iteration}*")]
            sublisted_eval_paths = create_eval_df_sublists(eval_paths)

            for eval_df_paths in sublisted_eval_paths:
                eval_df = concat_and_aggreagte_dfs(
                    df_paths=eval_df_paths,
                    idx_cols=idx_cols,
                    score_cols=score_cols
                )

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


def fixed_start_permutations(lst):
    """Generate fixed-start permutations (rolling behavior)."""
    for i in range(len(lst)):
        yield lst[i:] + lst[:i]


def generate_next_train_run(sample_files, mode="permutations"):
    if mode == "permutations":
        train_combinations = permutations(sample_files, len(sample_files))
    elif mode == "rolling":
        train_combinations = fixed_start_permutations(sample_files)
    else:
        raise ValueError("Invalid mode. Choose 'permutations' or 'rolling'.")

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


if __name__ == "__main__":
    dir = Path("../tests/_data")
    dir.mkdir(exist_ok=True)

    df = pd.DataFrame(
        {
            "spectrum_id": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "score": [1, 1, 1, 1, 1, 1, 1, 1],
            "is_decoy": [0, 0, 0, 0, 0, 0, 0, 1],
        }
    )

    generate_and_pickle_samples(
        df,
        sample_group_col="spectrum_id",
        n_samples=3,
        sample_dir=dir,
        file_extension="pkl",
    )

    gen = generate_next_train_run([p for p in dir.glob("*.pkl")])

    for i, train_sample_file, eval_sample_file, train_path_md5, finished_path in gen:
        print(i, train_sample_file, eval_sample_file, train_path_md5, finished_path)

        if finished_path:
            continue

        eval_df = pd.read_pickle(eval_sample_file)
        eval_data_md5 = get_idx_md5(eval_df["spectrum_id"].unique(), sort_ids=True)
        output_path = (
                dir
                / f"eval__path_{train_path_md5}__iteration_{i}__data_{eval_data_md5}.csv"
        )

        eval_df.to_csv(output_path, index=False)

    for p in dir.glob("*.pkl"):
        p.unlink()
