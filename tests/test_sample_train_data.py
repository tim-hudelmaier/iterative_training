import hashlib
from pathlib import Path

import pytest
import pandas as pd

from iter_helpers.iter_helpers import (
    drop_cols,
    generate_and_pickle_samples,
    get_idx_md5,
    generate_next_train_run,
    generate_eval_df,
    create_eval_df_sublists
)


def test_drop_cols():
    data = {"pre_col1": [1, 2], "pre_col2": [3, 4], "other_col": [5, 6]}
    df = pd.DataFrame(data)

    prefixes_to_drop = ["pre_"]
    modified_df = drop_cols(df.copy(), prefixes_to_drop)
    assert "pre_col1" not in modified_df.columns
    assert "pre_col2" not in modified_df.columns
    assert "other_col" in modified_df.columns

    modified_df = drop_cols(df.copy(), [])
    assert all(col in modified_df.columns for col in df.columns)

    # Test with a prefix not present in the DataFrame
    prefixes_to_drop = ["nonexistent_prefix_"]
    modified_df = drop_cols(df.copy(), prefixes_to_drop)
    assert all(col in modified_df.columns for col in df.columns)


def test_generate_and_pickle_samples():
    test_dir = Path("tests") / "test_samples"
    test_dir.mkdir(exist_ok=True)

    # Create a sample DataFrame
    data = {
        "sample_group_col": ["A", "B", "C", "A", "B", "C"],
        "value": [1, 2, 3, 4, 5, 6],
    }
    df = pd.DataFrame(data)

    n_samples = 2
    file_extension = "pkl"
    all_samples_md5 = generate_and_pickle_samples(
        df, "sample_group_col", n_samples, test_dir, file_extension
    )

    # Check if the correct number of files are created
    assert len(list(test_dir.glob(f"*.{file_extension}"))) == n_samples

    # Load pickled data and check MD5 hash
    concatenated_df = pd.concat(
        [pd.read_pickle(f) for f in test_dir.glob(f"*.{file_extension}")]
    )
    concatenated_md5 = get_idx_md5(
        concatenated_df["sample_group_col"].unique(), sort_ids=True
    )
    assert all_samples_md5 == concatenated_md5

    # Check that a unique group value is only in one file
    for value in df["sample_group_col"].unique():
        count = sum(
            value in pd.read_pickle(f)["sample_group_col"].values
            for f in test_dir.glob(f"*.{file_extension}")
        )
        assert count == 1

    for item in test_dir.iterdir():
        item.unlink()
    test_dir.rmdir()


def test_get_idx_md5_with_list():
    ids = ["id3", "id1", "id2"]
    sorted_ids = sorted(ids)
    assert get_idx_md5(ids) == hashlib.md5("".join(sorted_ids).encode()).hexdigest()


def test_get_idx_md5_with_list_and_duplicate():
    ids = ["id3", "id1", "id2", "id1"]
    sorted_ids = sorted(["id3", "id1", "id2"])
    assert get_idx_md5(ids) == hashlib.md5("".join(sorted_ids).encode()).hexdigest()


def test_get_idx_md5_with_series():
    ids = pd.Series(["id3", "id1", "id2", "id1"])  # Includes a duplicate
    unique_sorted_ids = sorted(ids.unique())
    assert (
            get_idx_md5(ids) == hashlib.md5(
        "".join(unique_sorted_ids).encode()).hexdigest()
    )


def test_get_idx_md5_with_string():
    id_str = "id1"
    assert get_idx_md5(id_str) == hashlib.md5(id_str.encode()).hexdigest()


def test_get_idx_md5_with_series_of_ints():
    ids = pd.Series([123, 456, 789, 123])  # Includes a duplicate
    unique_sorted_ids = sorted(ids.unique())
    unique_sorted_ids = [str(i) for i in unique_sorted_ids]
    assert (
            get_idx_md5(ids) == hashlib.md5(
        "".join(unique_sorted_ids).encode()).hexdigest()
    )


def test_get_idx_md5_with_list_of_ints():
    ids = [123, 456, 789, 123] # Includes a duplicate
    unique_sorted_ids = sorted(list(set(ids)))
    unique_sorted_ids = [str(i) for i in unique_sorted_ids]
    assert (
            get_idx_md5(ids) == hashlib.md5(
        "".join(unique_sorted_ids).encode()).hexdigest()
    )


def test_get_idx_md5_with_invalid_type():
    with pytest.raises(ValueError):
        get_idx_md5(123)  # Using an integer should raise a ValueError


def test_get_idx_md5_sorting_behavior():
    ids1 = ["id3", "id1", "id2"]
    ids2 = ["id2", "id1", "id3"]
    assert get_idx_md5(ids1, sort_ids=True) == get_idx_md5(ids2, sort_ids=True)
    assert get_idx_md5(ids1, sort_ids=False) != get_idx_md5(ids2, sort_ids=False)


@pytest.mark.parametrize(
    "sample_files, mode, expected_outputs",
    [
        (
                ["A", "B", "C"],
                "permutations",
                [
                    (0, "A", "B", get_idx_md5(["A", "B", "C"], sort_ids=False), False),
                    (1, "B", "C", get_idx_md5(["A", "B", "C"], sort_ids=False), False),
                    (2, "C", None, get_idx_md5(["A", "B", "C"], sort_ids=False), True),
                    (0, "A", "C", get_idx_md5(["A", "C", "B"], sort_ids=False), False),
                    (1, "C", "B", get_idx_md5(["A", "C", "B"], sort_ids=False), False),
                    (2, "B", None, get_idx_md5(["A", "C", "B"], sort_ids=False), True),
                    (0, "B", "A", get_idx_md5(["B", "A", "C"], sort_ids=False), False),
                    (1, "A", "C", get_idx_md5(["B", "A", "C"], sort_ids=False), False),
                    (2, "C", None, get_idx_md5(["B", "A", "C"], sort_ids=False), True),
                    (0, "B", "C", get_idx_md5(["B", "C", "A"], sort_ids=False), False),
                    (1, "C", "A", get_idx_md5(["B", "C", "A"], sort_ids=False), False),
                    (2, "A", None, get_idx_md5(["B", "C", "A"], sort_ids=False), True),
                    (0, "C", "A", get_idx_md5(["C", "A", "B"], sort_ids=False), False),
                    (1, "A", "B", get_idx_md5(["C", "A", "B"], sort_ids=False), False),
                    (2, "B", None, get_idx_md5(["C", "A", "B"], sort_ids=False), True),
                    (0, "C", "B", get_idx_md5(["C", "B", "A"], sort_ids=False), False),
                    (1, "B", "A", get_idx_md5(["C", "B", "A"], sort_ids=False), False),
                    (2, "A", None, get_idx_md5(["C", "B", "A"], sort_ids=False), True),
                ],
        ),
        (
                ["A", "B", "C"],
                "rolling",
                [
                    (0, "A", "B", get_idx_md5(["A", "B", "C"], sort_ids=False), False),
                    (1, "B", "C", get_idx_md5(["A", "B", "C"], sort_ids=False), False),
                    (2, "C", None, get_idx_md5(["A", "B", "C"], sort_ids=False), True),
                    (0, "B", "C", get_idx_md5(["B", "C", "A"], sort_ids=False), False),
                    (1, "C", "A", get_idx_md5(["B", "C", "A"], sort_ids=False), False),
                    (2, "A", None, get_idx_md5(["B", "C", "A"], sort_ids=False), True),
                    (0, "C", "A", get_idx_md5(["C", "A", "B"], sort_ids=False), False),
                    (1, "A", "B", get_idx_md5(["C", "A", "B"], sort_ids=False), False),
                    (2, "B", None, get_idx_md5(["C", "A", "B"], sort_ids=False), True),
                ],
        ),
    ],
)
def test_generate_next_train_run(sample_files, mode, expected_outputs):
    gen = generate_next_train_run(sample_files, mode=mode)

    for i, (idx, train_sample_file, eval_sample_file, md5, finished_path) in enumerate(
            gen
    ):
        assert (
                   idx,
                   train_sample_file,
                   eval_sample_file,
                   md5,
                   finished_path,
               ) == expected_outputs[i]


def test_generate_eval_df_consolidate_mode():
    test_dir = pytest._test_path / "_data"

    original_md5 = get_idx_md5(["A", "B", "C", "D"], sort_ids=True)

    iteration_1_df = pd.DataFrame(
        {
            "spectrum_id": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "score": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            "is_decoy": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    iteration_2_df = pd.DataFrame(
        {
            "spectrum_id": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "score": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "is_decoy": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    expected_results = [iteration_1_df, iteration_2_df]

    gen = generate_eval_df(n_iterations=2, evals_dir=test_dir, method="consolidate")

    for iteration, eval_df, eval_md5 in gen:
        assert eval_md5 == original_md5

        sorted_df = eval_df.sort_index(axis=1).sort_values(
            by=["spectrum_id"]).reset_index(drop=True)
        sorted_exp = expected_results[iteration].sort_index(axis=1).sort_values(
            by=["spectrum_id"]).reset_index(drop=True)

        assert sorted_df.equals(sorted_exp)


def test_generate_eval_df_rolling_mode():
    test_dir = pytest._test_path / "_data"

    original_md5 = get_idx_md5(["A", "B", "C", "D"], sort_ids=True)

    iteration_1_df_A = pd.DataFrame(
        {
            "spectrum_id": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "score": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "is_decoy": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    iteration_1_df_B = pd.DataFrame(
        {
            "spectrum_id": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "score": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "is_decoy": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    iteration_2_df = pd.DataFrame(
        {
            "spectrum_id": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "score": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "is_decoy": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    expected_results = [
        {
            "A": iteration_1_df_A.sort_index(axis=1).sort_values(
                by=["spectrum_id"]).reset_index(drop=True),
            "B": iteration_1_df_B.sort_index(axis=1).sort_values(
                by=["spectrum_id"]).reset_index(drop=True)
        },
        {
            "A": iteration_2_df.sort_index(axis=1).sort_values(
                by=["spectrum_id"]).reset_index(drop=True),
            "B": iteration_2_df.sort_index(axis=1).sort_values(
                by=["spectrum_id"]).reset_index(drop=True)
        }
    ]

    gen = generate_eval_df(n_iterations=2, evals_dir=test_dir, method="rolling")

    for iteration, eval_df, eval_md5 in gen:
        assert eval_md5 == original_md5

        sorted_df = eval_df.sort_index(axis=1).sort_values(
            by=["spectrum_id"]).reset_index(drop=True)

        assert sorted_df.equals(expected_results[iteration]["A"]) or sorted_df.equals(
            expected_results[iteration]["B"])

        if sorted_df.equals(expected_results[iteration]["A"]):
            expected_results[iteration]["A"] = None
        else:
            expected_results[iteration]["B"] = None


@pytest.mark.parametrize(
    "input_list",
    [
        (
                [
                    "something__path_1__data_a.pkl",
                    "something__path_2__data_b.pkl",
                    "something__path_3__data_c.pkl",
                    "something__path_4__data_a.pkl",
                    "something__path_5__data_b.pkl",
                    "something__path_6__data_c.pkl",
                ]
        ),
        (
                [
                    "something__path_1__data_a.pkl",
                    "something__path_2__data_b.pkl",
                    "something__path_3__data_c.pkl",
                    "something__path_4__data_a.pkl",
                    "something__path_5__data_b.pkl",
                    "something__path_6__data_c.pkl",
                ]
        ),
        (
                [
                    "something__path_1__data_a.pkl",
                    "something__path_2__data_b.pkl",
                    "something__path_3__data_c.pkl",
                    "something__path_4__data_a.pkl",
                    "something__path_5__data_b.pkl",
                    "something__path_6__data_c.pkl",
                    "something__path_7__data_a.pkl",
                    "something__path_8__data_b.pkl",
                ]
        ),

    ]
)
def test_eval_df_sublists(input_list):
    t = create_eval_df_sublists(input_list)
    assert create_eval_df_sublists(input_list) == [
        [
            "something__path_1__data_a.pkl",
            "something__path_2__data_b.pkl",
            "something__path_3__data_c.pkl"
        ],
        [
            "something__path_4__data_a.pkl",
            "something__path_5__data_b.pkl",
            "something__path_6__data_c.pkl",
        ]
    ]
