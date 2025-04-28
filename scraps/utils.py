"""
Utility functions for data manipulation and analysis.
"""

import logging
from typing import Dict, Optional, Set, Union

import pandas as pd

from napistu import mechanism_matching
from napistu.constants import ONTOLOGIES_LIST

logger = logging.getLogger(__name__)


def _validate_wide_ontologies(
    wide_df: pd.DataFrame,
    ontologies: Optional[Union[str, Set[str], Dict[str, str]]] = None
) -> Set[str]:
    """
    Validate ontology specifications against the wide DataFrame and ONTOLOGIES_LIST.
    
    Parameters
    ----------
    wide_df : pd.DataFrame
        DataFrame with one column per ontology and a results column
    ontologies : Optional[Union[str, Set[str], Dict[str, str]]]
        Either:
        - String specifying a single ontology column
        - Set of columns to treat as ontologies
        - Dict mapping wide column names to ontology names
        - None to automatically detect ontology columns based on ONTOLOGIES_LIST
        
    Returns
    -------
    Set[str]
        Set of validated ontology names. For dictionary mappings, returns the target ontology names.
        
    Raises
    ------
    ValueError
        If validation fails for any ontology specification or no valid ontologies are found
    """
    # Convert string input to set
    if isinstance(ontologies, str):
        ontologies = {ontologies}

    # Get the set of ontology columns
    if isinstance(ontologies, dict):
        # Check source columns exist in DataFrame
        missing_cols = set(ontologies.keys()) - set(wide_df.columns)
        if missing_cols:
            raise ValueError(
                f"Source columns not found in DataFrame: {missing_cols}"
            )
        # Validate target ontologies against ONTOLOGIES_LIST
        invalid_onts = set(ontologies.values()) - set(ONTOLOGIES_LIST)
        if invalid_onts:
            raise ValueError(
                f"Invalid ontologies in mapping: {invalid_onts}. Must be one of: {ONTOLOGIES_LIST}"
            )
        # Return target ontology names instead of source column names
        ontology_cols = set(ontologies.values())
        
    elif isinstance(ontologies, set):
        # Check specified columns exist in DataFrame
        missing_cols = ontologies - set(wide_df.columns)
        if missing_cols:
            raise ValueError(
                f"Specified ontology columns not found in DataFrame: {missing_cols}"
            )
        # Validate specified ontologies against ONTOLOGIES_LIST
        invalid_onts = ontologies - set(ONTOLOGIES_LIST)
        if invalid_onts:
            raise ValueError(
                f"Invalid ontologies in set: {invalid_onts}. Must be one of: {ONTOLOGIES_LIST}"
            )
        ontology_cols = ontologies
        
    else:
        # Auto-detect ontology columns by matching against ONTOLOGIES_LIST
        ontology_cols = set(wide_df.columns) & set(ONTOLOGIES_LIST)
        if not ontology_cols:
            raise ValueError(
                f"No valid ontology columns found in DataFrame. Column names must match one of: {ONTOLOGIES_LIST}"
            )
        logger.info(
            f"Auto-detected ontology columns: {ontology_cols}"
        )
    
    logger.debug(f"Validated ontology columns: {ontology_cols}")
    return ontology_cols


def match_features_to_pathway_species_wide(
    wide_df: pd.DataFrame,
    species_identifiers: pd.DataFrame,
    ontologies: Optional[Union[Set[str], Dict[str, str]]] = None,
    feature_id_var: str = "identifier"
) -> pd.DataFrame:
    """
    Convert a wide-format DataFrame with multiple ontology columns to long format,
    and call mechanism_matching.features_to_pathway_species.

    Parameters
    ----------
    wide_df : pd.DataFrame
        DataFrame with ontology identifier columns and any number of results columns.
        All non-ontology columns are treated as results.
    species_identifiers : pd.DataFrame
        DataFrame as required by features_to_pathway_species
    ontologies : Optional[Union[Set[str], Dict[str, str]]], default=None
        Either:
        - Set of columns to treat as ontologies
        - Dict mapping wide column names to ontology names
        - None to automatically detect ontology columns based on ONTOLOGIES_LIST
    feature_id_var : str, default="identifier"
        Name for the identifier column in the long format

    Returns
    -------
    pd.DataFrame
        Output of mechanism_matching.features_to_pathway_species

    Examples
    --------
    >>> # Example with auto-detected ontology columns and multiple results
    >>> wide_df = pd.DataFrame({
    ...     'uniprot': ['P12345', 'Q67890'],
    ...     'chebi': ['15377', '16810'],
    ...     'log2fc': [1.0, 2.0],
    ...     'pvalue': [0.01, 0.05]
    ... })
    >>> result = match_features_to_pathway_species_wide(
    ...     wide_df=wide_df,
    ...     species_identifiers=species_identifiers
    ... )

    >>> # Example with custom ontology mapping
    >>> wide_df = pd.DataFrame({
    ...     'protein_id': ['P12345', 'Q67890'],
    ...     'compound_id': ['15377', '16810'],
    ...     'expression': [1.0, 2.0],
    ...     'confidence': [0.8, 0.9]
    ... })
    >>> result = match_features_to_pathway_species_wide(
    ...     wide_df=wide_df,
    ...     species_identifiers=species_identifiers,
    ...     ontologies={'protein_id': 'uniprot', 'compound_id': 'chebi'}
    ... )
    """
    # Validate ontologies first
    ontology_cols = _validate_wide_ontologies(wide_df, ontologies)
    
    # Apply renaming if a mapping is provided
    if isinstance(ontologies, dict):
        wide_df = wide_df.rename(columns=ontologies)
        # Update ontology_cols to use the new names
        ontology_cols = {ontologies[col] for col in ontology_cols}

    # All non-ontology columns are treated as results
    results_cols = list(set(wide_df.columns) - ontology_cols)
    if not results_cols:
        raise ValueError("No results columns found in DataFrame")
    
    logger.info(f"Using columns as results: {results_cols}")

    # Create long format DataFrames for each results column
    results_dfs = []
    for results_col in results_cols:
        long_df = wide_df.melt(
            id_vars=[results_col],
            value_vars=list(ontology_cols),
            var_name="ontology",
            value_name=feature_id_var
        ).dropna(subset=[feature_id_var])
        
        # Rename results column to a standard name for mechanism_matching
        long_df = long_df.rename(columns={results_col: "results"})
        
        # Reorder columns
        long_df = long_df[["ontology", feature_id_var, "results"]]
        results_dfs.append(long_df)
        
        logger.debug(
            f"Melted DataFrame for {results_col} from {wide_df.shape} to {long_df.shape}"
        )

    # Combine all results
    final_df = pd.concat(results_dfs, axis=0, ignore_index=True)
    logger.debug(f"Combined DataFrame shape: {final_df.shape}")

    # Call the matching function
    return mechanism_matching.features_to_pathway_species(
        feature_identifiers=final_df,
        species_identifiers=species_identifiers,
        ontologies=ontology_cols,
        feature_id_var=feature_id_var
    ) 