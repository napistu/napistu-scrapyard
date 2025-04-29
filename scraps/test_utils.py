import pandas as pd

from utils import _drop_extra_cols
from utils import resolve_matches
from utils import _aggregate_numeric_columns

def test_drop_extra_cols():
    """Test the _drop_extra_cols function for removing and reordering columns."""
    # Setup test DataFrames
    df_in = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'col3': [7, 8, 9]
    })
    
    df_out = pd.DataFrame({
        'col2': [10, 11, 12],
        'col3': [13, 14, 15],
        'col4': [16, 17, 18],  # Extra column that should be dropped
        'col1': [19, 20, 21]   # Different order than df_in
    })
    
    # Call the function
    result = _drop_extra_cols(df_in, df_out)
    
    # Check that extra column was dropped
    assert 'col4' not in result.columns
    
    # Check that columns are in the same order as df_in
    assert list(result.columns) == list(df_in.columns)
    
    # Check that values are preserved
    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame({
            'col1': [19, 20, 21],
            'col2': [10, 11, 12],
            'col3': [13, 14, 15]
        })[list(df_in.columns)]  # Ensure same column order
    )
    
    # Test with no overlapping columns
    df_out_no_overlap = pd.DataFrame({
        'col4': [1, 2, 3],
        'col5': [4, 5, 6]
    })
    result_no_overlap = _drop_extra_cols(df_in, df_out_no_overlap)
    assert result_no_overlap.empty
    assert list(result_no_overlap.columns) == []
    
    # Test with subset of columns
    df_out_subset = pd.DataFrame({
        'col1': [1, 2, 3],
        'col3': [7, 8, 9]
    })
    result_subset = _drop_extra_cols(df_in, df_out_subset)
    assert list(result_subset.columns) == ['col1', 'col3']
    pd.testing.assert_frame_equal(result_subset, df_out_subset[['col1', 'col3']])


def test_resolve_matches_with_example_data():
    """Test resolve_matches function with example data for all aggregation methods."""
    # Setup example data with overlapping 1-to-many and many-to-1 cases
    example_data = pd.DataFrame({
        "feature_id": ["A", "B", "C", "D", "D", "E", "B", "B", "C"],
        "s_id": ["s_id_1", "s_id_1", "s_id_1", "s_id_4", "s_id_5", "s_id_6", "s_id_2", "s_id_3", "s_id_3"],
        "results_a": [1, 2, 3, 0.4, 5, 6, 0.7, 0.8, 9],
        "results_b": ["foo", "foo", "bar", "bar", "baz", "baz", "not", "not", "not"]
    })
    
    # Test with keep_id_col=True (default)
    result_with_id = resolve_matches(example_data, keep_id_col=True, numeric_agg="mean")
    
    # Verify feature_id column is present and correctly aggregated
    assert "feature_id" in result_with_id.columns
    assert result_with_id.loc["s_id_1", "feature_id"] == "A,B,C"
    assert result_with_id.loc["s_id_3", "feature_id"] == "B,C"
    
    # Test with keep_id_col=False
    result_without_id = resolve_matches(example_data, keep_id_col=False, numeric_agg="mean")
    
    # Verify feature_id column is not in output
    assert "feature_id" not in result_without_id.columns
    
    # Verify other columns are still present and correctly aggregated
    assert "results_a" in result_without_id.columns
    assert "results_b" in result_without_id.columns
    assert "feature_id_match_count" in result_without_id.columns
    
    # Verify numeric aggregation still works
    actual_mean = result_without_id.loc["s_id_1", "results_a"]
    expected_mean = 2.0  # (1 + 2 + 3) / 3
    assert actual_mean == expected_mean, f"Expected mean {expected_mean}, but got {actual_mean}"
    
    # Verify string aggregation still works
    assert result_without_id.loc["s_id_1", "results_b"] == "bar,foo"
    
    # Verify match counts are still present
    assert result_without_id.loc["s_id_1", "feature_id_match_count"] == 3
    assert result_without_id.loc["s_id_3", "feature_id_match_count"] == 2
    
    # Test weighted mean (feature_id is used for weights regardless of keep_id_col)
    weighted_result = resolve_matches(example_data, numeric_agg="weighted_mean", keep_id_col=True)
    
    # For s_id_1:
    # A appears once in total (weight = 1/1)
    # B appears three times in total (weight = 1/3)
    # C appears twice in total (weight = 1/2)
    # Sum of unnormalized weights = 1 + 1/3 + 1/2 = 1.833
    # Normalized weights:
    # A: (1/1)/1.833 = 0.545
    # B: (1/3)/1.833 = 0.182
    # C: (1/2)/1.833 = 0.273
    # Weighted mean = 1×0.545 + 2×0.182 + 3×0.273 = 1.73
    actual_weighted_mean_1 = weighted_result.loc["s_id_1", "results_a"]
    expected_weighted_mean_1 = 1.73
    assert abs(actual_weighted_mean_1 - expected_weighted_mean_1) < 0.01, \
        f"s_id_1 weighted mean: expected {expected_weighted_mean_1:.3f}, but got {actual_weighted_mean_1:.3f}"
    
    # For s_id_3:
    # B appears three times in total (weight = 1/3)
    # C appears twice in total (weight = 1/2)
    # Sum of unnormalized weights = 1/3 + 1/2 = 0.833
    # Normalized weights:
    # B: (1/3)/0.833 = 0.4
    # C: (1/2)/0.833 = 0.6
    # Weighted mean = 0.8×0.4 + 9×0.6 = 5.72
    actual_weighted_mean_3 = weighted_result.loc["s_id_3", "results_a"]
    expected_weighted_mean_3 = 5.72
    assert abs(actual_weighted_mean_3 - expected_weighted_mean_3) < 0.01, \
        f"s_id_3 weighted mean: expected {expected_weighted_mean_3:.3f}, but got {actual_weighted_mean_3:.3f}"
    
    # Test weighted mean with keep_id_col=False (weights still use feature_id)
    weighted_result_no_id = resolve_matches(example_data, numeric_agg="weighted_mean", keep_id_col=False)
    
    # Verify weighted means are the same regardless of keep_id_col
    assert abs(weighted_result_no_id.loc["s_id_1", "results_a"] - expected_weighted_mean_1) < 0.01, \
        "Weighted mean should be the same regardless of keep_id_col"
    assert abs(weighted_result_no_id.loc["s_id_3", "results_a"] - expected_weighted_mean_3) < 0.01, \
        "Weighted mean should be the same regardless of keep_id_col"
    
    # Test that both versions preserve the same index structure
    expected_index = pd.Index(["s_id_1", "s_id_2", "s_id_3", "s_id_4", "s_id_5", "s_id_6"])
    pd.testing.assert_index_equal(result_with_id.index, expected_index)
    pd.testing.assert_index_equal(result_without_id.index, expected_index)

def test_aggregate_numeric_basic():
    """Test basic aggregation methods in _aggregate_numeric_columns."""
    # Setup simple test data
    df = pd.DataFrame({
        "feature_id": ["A", "B", "B", "C"],
        "value1": [1.0, 2.0, 3.0, 4.0],
        "value2": [10.0, 20.0, 30.0, 40.0]
    })
    numeric_cols = ["value1", "value2"]
    
    # Test mean
    result_mean = _aggregate_numeric_columns(df, numeric_cols, method="mean")
    assert result_mean["value1"] == 2.5  # (1 + 2 + 3 + 4) / 4
    assert result_mean["value2"] == 25.0  # (10 + 20 + 30 + 40) / 4
    
    # Test max
    result_max = _aggregate_numeric_columns(df, numeric_cols, method="max")
    assert result_max["value1"] == 4.0
    assert result_max["value2"] == 40.0

def test_aggregate_numeric_weighted_mean():
    """Test weighted mean calculation in _aggregate_numeric_columns."""
    # Setup test data with known feature frequencies
    df = pd.DataFrame({
        "feature_id": ["A", "B", "B", "C"],  # B appears twice
        "value": [100.0, 1.0, 2.0, 200.0]
    })
    
    result = _aggregate_numeric_columns(df, ["value"], method="weighted_mean")
    
    # Expected weights:
    # A: 1/1 = 1.0
    # B: 1/2 = 0.5 (appears twice)
    # C: 1/1 = 1.0
    # Normalized weights:
    # Total = 1.0 + 0.5 + 0.5 + 1.0 = 3.0
    # A: 1.0/3.0 = 0.333
    # B: 0.5/3.0 = 0.167 (each B)
    # C: 1.0/3.0 = 0.333
    
    # Expected weighted mean:
    # (100 * 0.333) + (1 * 0.167) + (2 * 0.167) + (200 * 0.333) ≈ 100.5
    expected_weighted = 100.5
    
    # Regular mean would be:
    # (100 + 1 + 2 + 200) / 4 = 75.75
    expected_unweighted = 75.75
    
    assert abs(result["value"] - expected_weighted) < 0.01, \
        f"Expected weighted mean {expected_weighted:.2f}, but got {result['value']:.2f}. " \
        f"Note: unweighted mean would be {expected_unweighted:.2f}"
    
    # Verify this is different from regular mean
    regular_mean = df["value"].mean()
    assert abs(result["value"] - regular_mean) > 20, \
        "Weighted mean should be notably different from regular mean"

def test_aggregate_numeric_edge_cases():
    """Test edge cases for _aggregate_numeric_columns."""
    # Single row
    df_single = pd.DataFrame({
        "feature_id": ["A"],
        "value": [1.0]
    })
    result_single = _aggregate_numeric_columns(df_single, ["value"], method="weighted_mean")
    assert result_single["value"] == 1.0
    
    # All same feature_id
    df_same = pd.DataFrame({
        "feature_id": ["A", "A", "A"],
        "value": [1.0, 2.0, 3.0]
    })
    result_same = _aggregate_numeric_columns(df_same, ["value"], method="weighted_mean")
    assert result_same["value"] == 2.0  # Should be simple mean when all weights are equal
    
    # Multiple numeric columns
    df_multi = pd.DataFrame({
        "feature_id": ["A", "B", "B"],
        "value1": [1.0, 2.0, 3.0],
        "value2": [10.0, 20.0, 30.0]
    })
    result_multi = _aggregate_numeric_columns(df_multi, ["value1", "value2"], method="weighted_mean")
    # A: weight = 0.5, B: weight = 0.25 each
    EXPECTED_VALUE_1 = 1.0 * 0.5 + 2.0 * 0.25 + 3.0 * 0.25  # = 1.75
    EXPECTED_VALUE_2 = 10.0 * 0.5 + 20.0 * 0.25 + 30.0 * 0.25  # = 17.5
    assert abs(result_multi["value1"] - EXPECTED_VALUE_1) < 0.01
    assert abs(result_multi["value2"] - EXPECTED_VALUE_2) < 0.01