import pandas as pd
import pytest
import numpy as np
from datetime import datetime

def _drop_extra_cols(
    df_in : pd.DataFrame,
    df_out : pd.DataFrame
    ) -> pd.DataFrame:
    """Remove columns in df_out that are not in df_in and order columns based on df_in."""

    retained_cols = df_in.columns.intersection(df_out.columns)
    df_out = df_out.loc[:, retained_cols]
    return df_out

def get_numeric_aggregator(method: str = "weighted_mean", feature_id_var: str = "feature_id") -> callable:
    """
    Get aggregation function for numeric columns with various methods.
    
    Parameters
    ----------
    method : str, default="weighted_mean"
        Aggregation method to use:
        - "weighted_mean": weighted by inverse of feature_id frequency (default)
        - "mean": simple arithmetic mean
        - "first": first value after sorting by feature_id_var (requires feature_id_var)
        - "max": maximum value
    feature_id_var : str, default="feature_id"
        Name of the identifier column for sorting and weighting
        
    Returns
    -------
    callable
        Aggregation function to use with groupby
        
    Raises
    ------
    ValueError
        If method="first" and feature_id_var is not in DataFrame
    """
    def weighted_mean(series: pd.Series) -> float:
        # During groupby aggregation, series.index is the original DataFrame index
        # and series.name is the column name. We can use this to get feature_ids
        parent_df = series.index.to_frame()
        
        # If feature_id_var not in parent DataFrame, fall back to mean
        if feature_id_var not in parent_df.columns:
            return series.mean()
            
        # Count occurrences of each feature_id in the entire original DataFrame
        feature_counts = parent_df[feature_id_var].value_counts()
        
        # Calculate weights as 1/N where N is the count of each feature_id
        weights = 1 / feature_counts[parent_df[feature_id_var]].values
        
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
        
        # Calculate weighted mean using original values and normalized weights
        return (series.values * weights).sum()
    
    def first_by_id(series: pd.Series) -> float:
        # Sort by feature_id_var and take first value
        df = series.reset_index()
        # Require feature_id_var for first method
        if feature_id_var not in df.columns:
            raise ValueError(
                f"Column '{feature_id_var}' not found in DataFrame. "
                f"This column is required when using method='first'"
            )
        return df.sort_values(feature_id_var).iloc[0][series.name]
    
    aggregators = {
        "weighted_mean": weighted_mean,
        "mean": "mean",  # Use pandas built-in mean
        "first": first_by_id,
        "max": "max"     # Use pandas built-in max
    }
    
    if method not in aggregators:
        raise ValueError(f"Unknown aggregation method: {method}. Must be one of {list(aggregators.keys())}")
    
    return aggregators[method]

def _aggregate_numeric_columns(
    df: pd.DataFrame, 
    numeric_cols: list[str], 
    method: str = "weighted_mean", 
    feature_id_var: str = "feature_id"
) -> pd.Series:
    """
    Aggregate numeric columns with specified method.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing data to aggregate
    numeric_cols : list
        List of numeric column names to aggregate
    method : str, default="weighted_mean"
        Aggregation method to use
    feature_id_var : str, default="feature_id"
        Name of the identifier column for weighting
        
    Returns
    -------
    pd.Series
        Series with aggregated values for each numeric column
    """
    if method != "weighted_mean":
        # Use simple aggregation for non-weighted methods
        return df[numeric_cols].agg(method)
        
    # For weighted mean, calculate weights based on feature_id frequency
    feature_counts = df[feature_id_var].value_counts()
    weights = 1 / feature_counts[df[feature_id_var]].values
    weights = weights / weights.sum()
    
    # Calculate weighted mean for each numeric column
    result = pd.Series(index=numeric_cols, dtype=float)
    for col in numeric_cols:
        result[col] = (df[col].values * weights).sum()
    
    return result

def test_resolve_matches_missing_id():
    """Test that resolve_matches raises an error when id_col is missing."""
    # Setup data without feature_id column
    data_no_id = pd.DataFrame({
        "s_id": ["s_id_1", "s_id_1", "s_id_2"],
        "results_a": [1, 2, 3],
        "results_b": ["foo", "bar", "baz"]
    })
    
    # Should raise KeyError when trying to use weighted mean
    with pytest.raises(KeyError, match="feature_id"):
        resolve_matches(data_no_id, numeric_agg="weighted_mean")
    
    # Should work fine with other aggregation methods
    result = resolve_matches(data_no_id, numeric_agg="mean")
    assert result.loc["s_id_1", "results_a"] == 1.5  # (1 + 2) / 2
    assert result.loc["s_id_1", "results_b"] == "bar,foo"

def test_resolve_matches_invalid_dtypes():
    """Test that resolve_matches raises an error for unsupported dtypes."""
    # Setup data with boolean and datetime columns
    data = pd.DataFrame({
        "feature_id": ["A", "B", "B", "C"],
        "bool_col": [True, False, True, False],
        "datetime_col": [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
            datetime(2024, 1, 4)
        ],
        "s_id": ["s1", "s1", "s2", "s2"]
    })
    
    # Should raise TypeError for unsupported dtypes
    with pytest.raises(TypeError, match="Unsupported data types"):
        resolve_matches(data)

def resolve_matches(
    matched_data: pd.DataFrame, 
    feature_id_var: str = "feature_id", 
    index_col: str = "s_id",
    numeric_agg: str = "weighted_mean",
    keep_id_col: bool = True
) -> pd.DataFrame:
    """
    Resolve many-to-1 and 1-to-many matches in matched data.
    
    Parameters
    ----------
    matched_data : pd.DataFrame
        DataFrame containing matched data with columns:
        - feature_id_var: identifier column (e.g. feature_id)
        - index_col: index column (e.g. s_id)
        - other columns: data columns to be aggregated
    feature_id_var : str, default="feature_id"
        Name of the identifier column
    index_col : str, default="s_id"
        Name of the column to use as index
    numeric_agg : str, default="weighted_mean"
        Method to aggregate numeric columns:
        - "weighted_mean": weighted by inverse of feature_id frequency (default)
        - "mean": simple arithmetic mean
        - "first": first value after sorting by feature_id_var (requires feature_id_var)
        - "max": maximum value
    keep_id_col : bool, default=True
        Whether to keep and rollup the feature_id_var in the output.
        If False, feature_id_var will be dropped from the output.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with resolved matches:
        - Many-to-1: numeric columns are aggregated using specified method
        - 1-to-many: adds a count column showing number of matches
        - Index is set to index_col
        
    Raises
    ------
    KeyError
        If feature_id_var is not present in the DataFrame and numeric_agg="weighted_mean"
    TypeError
        If DataFrame contains unsupported data types (boolean or datetime)
    """
    # Make a copy to avoid modifying input
    df = matched_data.copy()
    
    # Check for unsupported dtypes
    unsupported_dtypes = df.select_dtypes(include=['bool', 'datetime64']).columns
    if not unsupported_dtypes.empty:
        raise TypeError(
            f"Unsupported data types found in columns: {list(unsupported_dtypes)}. "
            "Boolean and datetime columns are not supported."
        )
    
    # Check if feature_id_var exists when using weighted mean
    if numeric_agg == "weighted_mean" and feature_id_var not in df.columns:
        raise KeyError(
            f"Column '{feature_id_var}' not found in DataFrame. "
            f"This column is required when using numeric_agg='weighted_mean'"
        )
    
    # Calculate global feature frequencies before setting index
    if numeric_agg == "weighted_mean":
        global_feature_counts = df[feature_id_var].value_counts()
    
    # Set index for grouping
    df = df.set_index(index_col)
    
    # Split columns by type
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
    if not keep_id_col:
        non_numeric_cols = non_numeric_cols.difference([feature_id_var])
    
    results = []
    
    # Handle non-numeric columns
    if len(non_numeric_cols) > 0:
        non_numeric_agg = df[non_numeric_cols].groupby(index_col).agg(
            lambda x: ','.join(sorted(set(x.astype(str))))
        )
        results.append(non_numeric_agg)
    
    # Handle numeric columns
    if len(numeric_cols) > 0:
        numeric_results = {}
        for group_idx, group_df in df.groupby(index_col):
            if numeric_agg == "weighted_mean":
                # Calculate weights based on global feature frequencies
                weights = 1 / global_feature_counts[group_df[feature_id_var]].values
                weights = weights / weights.sum()
                
                # Calculate weighted mean for each numeric column
                group_result = pd.Series(index=numeric_cols, dtype=float)
                for col in numeric_cols:
                    group_result[col] = (group_df[col].values * weights).sum()
            else:
                # Use simple aggregation for non-weighted methods
                group_result = group_df[numeric_cols].agg(numeric_agg)
            
            numeric_results[group_idx] = group_result
        
        # Convert numeric results to DataFrame with proper index
        numeric_agg = pd.DataFrame.from_dict(numeric_results, orient='index')
        results.append(numeric_agg)
    
    # Combine results
    if results:
        resolved = pd.concat(results, axis=1)
    else:
        resolved = pd.DataFrame(index=df.index)
    
    # Add count of matches per feature_id
    if feature_id_var in df.columns:
        match_counts = df.groupby(index_col)[feature_id_var].nunique()
        resolved[f'{feature_id_var}_match_count'] = match_counts
    
    # Drop feature_id_var if not keeping it
    if not keep_id_col and feature_id_var in resolved.columns:
        resolved = resolved.drop(columns=[feature_id_var])
    
    return resolved
