import pandas as pd
import pytest
import numpy as np
from datetime import datetime

from napistu import sbml_dfs_core
from napistu import sbml_dfs_utils
from napistu.sbml_dfs_utils import _validate_assets_sbml_ids
from napistu.mechanism_matching import _check_species_identifiers_table
from napistu.constants import IDENTIFIERS
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _log_feature_species_mapping_stats(pathway_species: pd.DataFrame):
    """
    Log statistics about the mapping between feature_id and s_id in the pathway_species DataFrame.
    """
    
    # Percent of feature_ids present one or more times in the output
    n_feature_ids = pathway_species['feature_id'].nunique()
    n_input_feature_ids = pathway_species['feature_id'].max() + 1 if 'feature_id' in pathway_species.columns else 0
    percent_present = 100 * n_feature_ids / n_input_feature_ids if n_input_feature_ids else 0
    logger.info(f"{percent_present:.1f}% of feature_ids are present one or more times in the output ({n_feature_ids}/{n_input_feature_ids})")

    # Number of times an s_id maps to 1+ feature_ids (with s_name)
    s_id_counts = pathway_species.groupby('s_id')['feature_id'].nunique()
    s_id_multi = s_id_counts[s_id_counts > 1]
    logger.info(f"{len(s_id_multi)} s_id(s) map to more than one feature_id.")
    if not s_id_multi.empty:
        examples = pathway_species[pathway_species['s_id'].isin(s_id_multi.index)][['s_id', 's_name', 'feature_id']]
        logger.info(f"Examples of s_id mapping to multiple feature_ids (showing up to 3):\n{examples.groupby(['s_id', 's_name'])['feature_id'].apply(list).head(3)}")

    # Number of times a feature_id maps to 1+ s_ids (with s_name)
    feature_id_counts = pathway_species.groupby('feature_id')['s_id'].nunique()
    feature_id_multi = feature_id_counts[feature_id_counts > 1]
    logger.info(f"{len(feature_id_multi)} feature_id(s) map to more than one s_id.")
    if not feature_id_multi.empty:
        examples = pathway_species[pathway_species['feature_id'].isin(feature_id_multi.index)][['feature_id', 's_id', 's_name']]
        logger.info(f"Examples of feature_id mapping to multiple s_ids (showing up to 3):\n{examples.groupby(['feature_id'])[['s_id', 's_name']].apply(lambda df: list(df.itertuples(index=False, name=None))).head(3)}")


def features_to_pathway_species(
    feature_identifiers: pd.DataFrame,
    species_identifiers: pd.DataFrame,
    ontologies: set,
    feature_identifiers_var: str,
    expand_identifiers: bool = False,
    identifier_delimiter: str = "/",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Features to Pathway Species

    Match a table of molecular species to their corresponding species in a pathway representation.

    Parameters:
    feature_identifiers: pd.DataFrame
        pd.Dataframe containing a "feature_identifiers_var" variable used to match entries
    species_identifiers: pd.DataFrame
        A table of molecular species identifiers produced from sbml_dfs.get_identifiers("species")
        generally using sbml_dfs_core.export_sbml_dfs()
    ontologies: set
        A set of ontologies used to match features to pathway species
    feature_identifiers_var: str
        Variable in "feature_identifiers" containing identifiers
    expand_identifiers: bool, default=False
        If True, split identifiers in feature_identifiers_var by identifier_delimiter and explode into multiple rows
    identifier_delimiter: str, default="/"
        Delimiter to use for splitting identifiers if expand_identifiers is True
    verbose: bool, default=False
        If True, log mapping statistics at the end of the function

    Returns:
    pathway_species: pd.DataFrame
        species_identifiers joined to feature_identifiers based on shared identifiers
    """

    # Check for identifier column
    if feature_identifiers_var not in feature_identifiers.columns.to_list():
        raise ValueError(
            f"{feature_identifiers_var} must be a variable in 'feature_identifiers', "
            f"possible variables are {', '.join(feature_identifiers.columns.tolist())}"
        )

    # Respect or create feature_id column
    if "feature_id" not in feature_identifiers.columns:
        logger.warning("No feature_id column found in feature_identifiers, creating one")
        feature_identifiers = feature_identifiers.copy()
        feature_identifiers["feature_id"] = np.arange(len(feature_identifiers))

    # Optionally expand identifiers into multiple rows
    if expand_identifiers:
        # Count the number of expansions by counting delimiters
        n_expansions = feature_identifiers[feature_identifiers_var].astype(str).str.count(identifier_delimiter).sum()
        if n_expansions > 0:
            logger.info(f"Expanding identifiers: {n_expansions} delimiters found in '{feature_identifiers_var}', will expand to more rows.")

        # Split, strip whitespace, and explode
        feature_identifiers = feature_identifiers.copy()
        feature_identifiers[feature_identifiers_var] = (
            feature_identifiers[feature_identifiers_var]
            .astype(str)
            .str.split(identifier_delimiter)
            .apply(lambda lst: [x.strip() for x in lst])
        )
        feature_identifiers = feature_identifiers.explode(feature_identifiers_var, ignore_index=True)

    # check identifiers table
    _check_species_identifiers_table(species_identifiers)

    available_ontologies = set(species_identifiers[IDENTIFIERS.ONTOLOGY].tolist())
    unavailable_ontologies = ontologies.difference(available_ontologies)

    # no ontologies present
    if len(unavailable_ontologies) == len(ontologies):
        raise ValueError(
            f"None of the requested ontologies ({', '.join(ontologies)}) "
            "were used to annotate pathway species. Available ontologies are: "
            f"{', '.join(available_ontologies)}"
        )

    # 1+ desired ontologies are not present
    if len(unavailable_ontologies) > 0:
        raise ValueError(
            f"Some of the requested ontologies ({', '.join(unavailable_ontologies)}) "
            "were NOT used to annotate pathway species. Available ontologies are: "
            f"{', '.join(available_ontologies)}"
        )

    relevant_identifiers = species_identifiers[
        species_identifiers[IDENTIFIERS.ONTOLOGY].isin(ontologies)
    ]

    # map features to pathway species
    pathway_species = feature_identifiers.merge(
        relevant_identifiers, left_on=feature_identifiers_var, right_on=IDENTIFIERS.IDENTIFIER
    )

    if pathway_species.shape[0] == 0:
        logger.warning(
            "None of the provided species identifiers matched entries of the pathway; returning None"
        )
        None

    # report the fraction of unmapped species
    if verbose:
        _log_feature_species_mapping_stats(pathway_species)

    return pathway_species


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


def _prepare_species_identifiers(
    sbml_dfs : sbml_dfs_core.SBML_dfs,
    dogmatic : bool = False,
    species_identifiers : Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Accepts and validates species_identifiers, or extracts a fresh table if None."""

    if species_identifiers is None:
        species_identifiers = sbml_dfs_utils.get_characteristic_species_ids(sbml_dfs, dogmatic = dogmatic)
    else:
        # check for compatibility
        try:
            # check species_identifiers format
            
            _check_species_identifiers_table(species_identifiers)
            # quick check for compatibility between sbml_dfs and species_identifiers
            _validate_assets_sbml_ids(sbml_dfs, species_identifiers)
        except ValueError as e:
            logger.warning(f"The provided identifiers are not compatible with your `sbml_dfs` object. Extracting a fresh species identifier table. {e}")
            species_identifiers = sbml_dfs_utils.get_characteristic_species_ids(sbml_dfs, dogmatic = dogmatic)

    return species_identifiers