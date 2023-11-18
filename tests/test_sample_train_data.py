import pytest

import pandas as pd

from iter_helpers.iter_helpers import drop_cols

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
