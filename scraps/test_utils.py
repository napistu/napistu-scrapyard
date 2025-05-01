import numpy as np
import pandas as pd
import pytest

import utils

from napistu.constants import SBML_DFS
from napistu.constants import IDENTIFIERS
from napistu.constants import ONTOLOGIES
from napistu.constants import FEATURE_ID_VAR_DEFAULT


def test_bind_wide_results(sbml_dfs):
    """
    Test that bind_wide_results correctly matches identifiers and adds results to species data.
    """
    # Get species identifiers, excluding reactome
    species_identifiers = (sbml_dfs
                          .get_identifiers(SBML_DFS.SPECIES)
                          .query("bqb == 'BQB_IS'")
                          .query("ontology != 'reactome'"))

    # Create example data with identifiers and results
    example_data = species_identifiers.groupby("ontology").head(10)[[IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER]]
    example_data["results_a"] = np.random.randn(len(example_data))
    example_data["results_b"] = np.random.randn(len(example_data))
    example_data[FEATURE_ID_VAR_DEFAULT] = range(0, len(example_data))

    # Create wide format data
    example_data_wide = (example_data
                        .pivot(columns=IDENTIFIERS.ONTOLOGY, 
                              values=IDENTIFIERS.IDENTIFIER, 
                              index=[FEATURE_ID_VAR_DEFAULT, "results_a", "results_b"])
                        .reset_index()
                        .rename_axis(None, axis=1))

    # Call bind_wide_results
    results_name = "test_results"
    sbml_dfs_result = utils.bind_wide_results(
        sbml_dfs=sbml_dfs,
        results_df=example_data_wide,
        results_name=results_name,
        ontologies={ONTOLOGIES.UNIPROT, ONTOLOGIES.CHEBI},
        dogmatic=False,
        species_identifiers=None,
        feature_id_var="feature_id",
        verbose=True
    )

    # Verify the results were added correctly
    assert results_name in sbml_dfs_result.species_data, f"{results_name} not found in species_data"
    
    # Get the bound results
    bound_results = sbml_dfs_result.species_data[results_name]

    # columns are feature_id, results_a, results_b
    assert set(bound_results.columns) == {FEATURE_ID_VAR_DEFAULT, "results_a", "results_b"}

    
