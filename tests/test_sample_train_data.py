from pathlib import Path

import pytest
import pandas as pd

from iter_helpers.iter_helpers import (
    drop_cols,
    generate_and_pickle_samples,
    get_idx_md5,
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
