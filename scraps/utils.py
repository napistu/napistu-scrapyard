import pandas as pd
import pytest
import numpy as np
from datetime import datetime

import logging

logger = logging.getLogger(__name__)

from types import SimpleNamespace

from napistu.constants import SBML_DFS

RESOLVE_MATCHES_AGGREGATORS = SimpleNamespace(
    WEIGHTED_MEAN = "weighted_mean",
    MEAN = "mean",
    FIRST = "first",
    MAX = "max"
)

RESOLVE_MATCHES_TMP_WEIGHT_COL = "__tmp_weight_for_aggregation__"
FEATURE_ID_VAR_DEFAULT = "feature_id"

def _drop_extra_cols(
    df_in : pd.DataFrame,
    df_out : pd.DataFrame
    ) -> pd.DataFrame:
    """Remove columns in df_out that are not in df_in and order columns based on df_in."""

    retained_cols = df_in.columns.intersection(df_out.columns)
    df_out = df_out.loc[:, retained_cols]
    return df_out

def get_numeric_aggregator(
    method: str = RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN, 
    feature_id_var: str = FEATURE_ID_VAR_DEFAULT,
) -> callable:
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
        Name of the column specifying a measured feature - used for sorting and weighting
        
    Returns
    -------
    callable
        Aggregation function to use with groupby
        
    Raises
    ------
    ValueError
        If method is not recognized
    """
    def weighted_mean(df: pd.DataFrame) -> float:
        # Get values and weights for this group
        values = df['value']
        weights = df['weight']
        # Weights are already normalized globally, just use them directly
        return (values * weights).sum() / weights.sum()
    
    def first_by_id(df: pd.DataFrame) -> float:
        # Sort by feature_id and take first value
        return df.sort_values(feature_id_var).iloc[0]['value']
    
    def simple_mean(series: pd.Series) -> float:
        return series.mean()
    
    def simple_max(series: pd.Series) -> float:
        return series.max()
    
    aggregators = {
        RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN: weighted_mean,
        RESOLVE_MATCHES_AGGREGATORS.MEAN: simple_mean,
        RESOLVE_MATCHES_AGGREGATORS.FIRST: first_by_id,
        RESOLVE_MATCHES_AGGREGATORS.MAX: simple_max
    }
    
    if method not in aggregators:
        raise ValueError(f"Unknown aggregation method: {method}. Must be one of {list(aggregators.keys())}")
    
    return aggregators[method]


def _split_numeric_non_numeric_columns(df: pd.DataFrame, always_non_numeric=None):
    """
    Utility to split DataFrame columns into numeric and non-numeric, always treating specified columns as non-numeric.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to split.
    always_non_numeric : list or set, optional
        Columns to always treat as non-numeric (e.g., ['feature_id']).
    
    Returns
    -------
    numeric_cols : pd.Index
        Columns considered numeric (int64, float64, and not in always_non_numeric).
    non_numeric_cols : pd.Index
        Columns considered non-numeric (object, string, etc., plus always_non_numeric).
    """
    if always_non_numeric is None:
        always_non_numeric = []
    always_non_numeric = set(always_non_numeric)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference(always_non_numeric)
    non_numeric_cols = df.columns.difference(numeric_cols)
    return numeric_cols, non_numeric_cols


def _aggregate_grouped_columns(
    df: pd.DataFrame,
    numeric_cols,
    non_numeric_cols,
    numeric_aggregator,
    feature_id_var: str = FEATURE_ID_VAR_DEFAULT,
    numeric_agg: str = RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN
) -> pd.DataFrame:
    """
    Aggregate numeric and non-numeric columns for grouped DataFrame.
    Assumes deduplication by feature_id within each s_id has already been performed.
    Returns the combined DataFrame.
    """
    results = []
    
    # Handle non-numeric columns
    if len(non_numeric_cols) > 0:
        non_numeric_agg = df[non_numeric_cols].groupby(level=0).agg(
            lambda x: ','.join(sorted(set(x.astype(str))))
        )
        results.append(non_numeric_agg)
    # Handle numeric columns
    if len(numeric_cols) > 0:
        numeric_results = {}
        for col in numeric_cols:
            if numeric_agg in [RESOLVE_MATCHES_AGGREGATORS.FIRST, RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN]:
                agg_df = pd.DataFrame({
                    'value': df[col],
                    feature_id_var: df[feature_id_var]
                })
                if numeric_agg == RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN:
                    agg_df[RESOLVE_MATCHES_TMP_WEIGHT_COL] = df[RESOLVE_MATCHES_TMP_WEIGHT_COL]
                numeric_results[col] = agg_df.groupby(level=0).apply(
                    lambda x: numeric_aggregator(x) if numeric_agg != RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN else numeric_aggregator(x.rename(columns={RESOLVE_MATCHES_TMP_WEIGHT_COL: 'weight'}))
                )
            else:
                numeric_results[col] = df[col].groupby(level=0).agg(numeric_aggregator)
        numeric_agg_df = pd.DataFrame(numeric_results)
        results.append(numeric_agg_df)
    # Combine results
    if results:
        resolved = pd.concat(results, axis=1)
    else:
        resolved = pd.DataFrame(index=df.index)
    return resolved


def resolve_matches(
    matched_data: pd.DataFrame, 
    feature_id_var: str = FEATURE_ID_VAR_DEFAULT, 
    index_col: str = SBML_DFS.S_ID,
    numeric_agg: str = RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN,
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
        - Index is set to index_col and named accordingly
        
    Raises
    ------
    KeyError
        If feature_id_var is not present in the DataFrame
    TypeError
        If DataFrame contains unsupported data types (boolean or datetime)
    """
    # Make a copy to avoid modifying input
    df = matched_data.copy()
    
    # Check for unsupported data types
    unsupported_dtypes = df.select_dtypes(include=['bool', 'datetime64']).columns
    if not unsupported_dtypes.empty:
        raise TypeError(f"Unsupported data types found in columns: {list(unsupported_dtypes)}. "
                       "Boolean and datetime columns are not supported.")
    
    # Always require feature_id_var
    if feature_id_var not in df.columns:
        raise KeyError(feature_id_var)
    
    # Deduplicate by feature_id within each s_id using groupby and first BEFORE any further processing
    df = df.groupby([index_col, feature_id_var], sort=False).first().reset_index()
    
    # Use a unique temporary column name for weights
    if RESOLVE_MATCHES_TMP_WEIGHT_COL in df.columns:
        raise ValueError(f"Temporary weight column name '{RESOLVE_MATCHES_TMP_WEIGHT_COL}' already exists in the input data. Please rename or remove this column and try again.")
    
    # Calculate weights if needed (after deduplication!)
    if numeric_agg == RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN:
        feature_counts = df[feature_id_var].value_counts()
        df[RESOLVE_MATCHES_TMP_WEIGHT_COL] = 1 / feature_counts[df[feature_id_var]].values
    
    # Set index for grouping
    df = df.set_index(index_col)
    
    # Use utility to split columns
    always_non_numeric = [feature_id_var] if keep_id_col else []
    numeric_cols, non_numeric_cols = _split_numeric_non_numeric_columns(df, always_non_numeric=always_non_numeric)
    
    # Get aggregator function
    numeric_aggregator = get_numeric_aggregator(
        method=numeric_agg, 
        feature_id_var=feature_id_var
    )
    resolved = _aggregate_grouped_columns(
        df,
        numeric_cols,
        non_numeric_cols,
        numeric_aggregator,
        feature_id_var=feature_id_var,
        numeric_agg=numeric_agg
    )
    # Add count of matches per feature_id
    match_counts = matched_data.groupby(index_col)[feature_id_var].nunique()
    resolved[f'{feature_id_var}_match_count'] = match_counts
    
    # Drop feature_id_var if not keeping it
    if not keep_id_col and feature_id_var in resolved.columns:
        resolved = resolved.drop(columns=[feature_id_var])
    
    # Ensure index is named consistently
    resolved.index.name = index_col
    
    return resolved
