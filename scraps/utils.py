"""
Utility functions for data manipulation and analysis.
"""

import logging
from typing import Dict, List, Optional, Set, Union

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


def match_features_to_wide_pathway_species(
    wide_df: pd.DataFrame,
    species_identifiers: pd.DataFrame,
    ontologies: Optional[Union[Set[str], Dict[str, str]]] = None,
    feature_id_var: str = "identifier"
) -> pd.DataFrame:
    """
    Convert a wide-format DataFrame with multiple ontology columns to long format,
    and match features to pathway species by ontology and identifier.

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
        Output of match_by_ontology_and_identifier

    Examples
    --------
    >>> # Example with auto-detected ontology columns and multiple results
    >>> wide_df = pd.DataFrame({
    ...     'uniprot': ['P12345', 'Q67890'],
    ...     'chebi': ['15377', '16810'],
    ...     'log2fc': [1.0, 2.0],
    ...     'pvalue': [0.01, 0.05]
    ... })
    >>> result = match_features_to_wide_pathway_species(
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
    >>> result = match_features_to_wide_pathway_species(
    ...     wide_df=wide_df,
    ...     species_identifiers=species_identifiers,
    ...     ontologies={'protein_id': 'uniprot', 'compound_id': 'chebi'}
    ... )
    """
    # Make a copy to avoid modifying the input
    wide_df = wide_df.copy()

    # Validate ontologies and get the set of ontology columns
    ontology_cols = _validate_wide_ontologies(wide_df, ontologies)
    melt_cols = list(ontology_cols)

    # Apply renaming if a mapping is provided
    if isinstance(ontologies, dict):
        wide_df = wide_df.rename(columns=ontologies)

    # All non-ontology columns are treated as results
    results_cols = list(set(wide_df.columns) - set(melt_cols))
    if not results_cols:
        raise ValueError("No results columns found in DataFrame")
    
    logger.info(f"Using columns as results: {results_cols}")

    # Melt ontology columns to long format, keeping all results columns
    long_df = wide_df.melt(
        id_vars=results_cols,
        value_vars=melt_cols,
        var_name="ontology",
        value_name=feature_id_var
    ).dropna(subset=[feature_id_var])    

    logger.debug(f"Final long format shape: {long_df.shape}")

    # Call the matching function with the validated ontologies
    return match_by_ontology_and_identifier(
        feature_identifiers=long_df,
        species_identifiers=species_identifiers,
        ontologies=ontology_cols,
        feature_id_var=feature_id_var
    )


def match_by_ontology_and_identifier(
    feature_identifiers: pd.DataFrame,
    species_identifiers: pd.DataFrame,
    ontologies: Union[str, Set[str], List[str]],
    feature_id_var: str = "identifier",
) -> pd.DataFrame:
    """
    Match features to pathway species based on both ontology and identifier matches.
    Performs separate matching for each ontology and concatenates the results.

    Parameters
    ----------
    feature_identifiers : pd.DataFrame
        DataFrame containing feature identifiers and results.
        Must have columns [ontology, feature_id_var, results]
    species_identifiers : pd.DataFrame
        DataFrame containing species identifiers from pathway.
        Must have columns [ontology, identifier]
    ontologies : Union[str, Set[str], List[str]]
        Ontologies to match on. Can be:
        - A single ontology string
        - A set of ontology strings
        - A list of ontology strings
    feature_id_var : str, default="identifier"
        Name of the identifier column in feature_identifiers

    Returns
    -------
    pd.DataFrame
        Concatenated results of matching for each ontology.
        Contains all columns from features_to_pathway_species()

    Examples
    --------
    >>> # Match using a single ontology
    >>> result = match_by_ontology_and_identifier(
    ...     feature_identifiers=features_df,
    ...     species_identifiers=species_df,
    ...     ontologies="uniprot"
    ... )

    >>> # Match using multiple ontologies
    >>> result = match_by_ontology_and_identifier(
    ...     feature_identifiers=features_df,
    ...     species_identifiers=species_df,
    ...     ontologies={"uniprot", "chebi"}
    ... )
    """
    # Convert string to set for consistent handling
    if isinstance(ontologies, str):
        ontologies = {ontologies}
    elif isinstance(ontologies, list):
        ontologies = set(ontologies)

    # Validate ontologies
    invalid_onts = ontologies - set(ONTOLOGIES_LIST)
    if invalid_onts:
        raise ValueError(
            f"Invalid ontologies specified: {invalid_onts}. Must be one of: {ONTOLOGIES_LIST}"
        )

    # Initialize list to store results
    matched_dfs = []

    # Process each ontology separately
    for ont in ontologies:
        # Filter feature identifiers to current ontology and drop ontology column
        ont_features = feature_identifiers[
            feature_identifiers["ontology"] == ont
        ].drop(columns=["ontology"]).copy()

        if ont_features.empty:
            logger.warning(f"No features found for ontology: {ont}")
            continue

        # Filter species identifiers to current ontology
        ont_species = species_identifiers[
            species_identifiers["ontology"] == ont
        ].copy()

        if ont_species.empty:
            logger.warning(f"No species found for ontology: {ont}")
            continue

        logger.debug(
            f"Matching {len(ont_features)} features to {len(ont_species)} species for ontology {ont}"
        )

        # Match features to species for this ontology
        matched = mechanism_matching.features_to_pathway_species(
            feature_identifiers=ont_features,
            species_identifiers=ont_species,
            ontologies={ont},
            feature_id_var=feature_id_var
        )

        if matched.empty:
            logger.warning(f"No matches found for ontology: {ont}")
            continue

        matched_dfs.append(matched)

    if not matched_dfs:
        logger.warning("No matches found for any ontology")
        return pd.DataFrame()  # Return empty DataFrame with correct columns

    # Combine results from all ontologies
    result = pd.concat(matched_dfs, axis=0, ignore_index=True)
    
    logger.info(
        f"Found {len(result)} total matches across {len(matched_dfs)} ontologies"
    )
    
    return result 