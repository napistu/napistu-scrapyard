import pytest
import pandas as pd

from utils import _validate_wide_ontologies
from utils import match_by_ontology_and_identifier

def test_validate_wide_ontologies():
    """Test the _validate_wide_ontologies function with various input types and error cases."""
    # Setup test data
    example_data_wide = pd.DataFrame({
        'results': [-1.0, 0.0, 1.0],
        'chebi': ['15377', '16810', '17925'],
        'uniprot': ['P12345', 'Q67890', 'O43826']
    })

    # Test auto-detection of ontology columns
    assert _validate_wide_ontologies(example_data_wide) == {"chebi", "uniprot"}

    # Test string input
    assert _validate_wide_ontologies(example_data_wide, ontologies="chebi") == {"chebi"}

    # Test set input
    assert _validate_wide_ontologies(example_data_wide, ontologies={"chebi"}) == {"chebi"}
    assert _validate_wide_ontologies(example_data_wide, ontologies={"chebi", "uniprot"}) == {"chebi", "uniprot"}

    # Test dictionary mapping for renaming
    assert _validate_wide_ontologies(
        example_data_wide, 
        ontologies={"chebi": "reactome", "uniprot": "ensembl_gene"}
    ) == {"reactome", "ensembl_gene"}

    # Test error cases
    
    # Missing column in set input (checks existence first)
    with pytest.raises(ValueError, match="Specified ontology columns not found in DataFrame:.*"):
        _validate_wide_ontologies(example_data_wide, ontologies={"invalid_ontology"})

    # Valid column name but invalid ontology
    df_with_invalid = pd.DataFrame({
        'results': [-1.0, 0.0, 1.0],
        'invalid_ontology': ['a', 'b', 'c'],
    })
    with pytest.raises(ValueError, match="Invalid ontologies in set:.*"):
        _validate_wide_ontologies(df_with_invalid, ontologies={"invalid_ontology"})

    # Missing source column in mapping
    with pytest.raises(ValueError, match="Source columns not found in DataFrame:.*"):
        _validate_wide_ontologies(example_data_wide, ontologies={"missing_column": "reactome"})

    # Invalid target ontology in mapping
    with pytest.raises(ValueError, match="Invalid ontologies in mapping:.*"):
        _validate_wide_ontologies(example_data_wide, ontologies={"chebi": "invalid_ontology"})

    # DataFrame with no valid ontology columns
    invalid_df = pd.DataFrame({
        'results': [-1.0, 0.0, 1.0],
        'col1': ['a', 'b', 'c'],
        'col2': ['d', 'e', 'f']
    })
    with pytest.raises(ValueError, match="No valid ontology columns found in DataFrame.*"):
        _validate_wide_ontologies(invalid_df)

def test_match_by_ontology_and_identifier():
    """Test the match_by_ontology_and_identifier function with various input types."""
    # Setup test data
    feature_identifiers = pd.DataFrame({
        "ontology": ["chebi", "chebi", "uniprot", "uniprot", "reactome"],
        "identifier": ["15377", "16810", "P12345", "Q67890", "R12345"],
        "results": [1.0, 2.0, -1.0, -2.0, 0.5]
    })

    species_identifiers = pd.DataFrame({
        "ontology": ["chebi", "chebi", "uniprot", "uniprot", "ensembl_gene"],
        "identifier": ["15377", "17925", "P12345", "O43826", "ENSG123"],
        "s_id": ["s1", "s2", "s3", "s4", "s5"],
        "s_name": ["compound1", "compound2", "protein1", "protein2", "gene1"],
        "bqb": ["BQB_IS"] * 5  # Add required bqb column with BQB_IS values
    })

    # Test with single ontology (string)
    result = match_by_ontology_and_identifier(
        feature_identifiers=feature_identifiers,
        species_identifiers=species_identifiers,
        ontologies="chebi"
    )
    assert len(result) == 1  # Only one matching chebi identifier
    assert result.iloc[0]["identifier"] == "15377"
    assert result.iloc[0]["results"] == 1.0
    assert result.iloc[0]["ontology"] == "chebi"  # From species_identifiers
    assert result.iloc[0]["s_name"] == "compound1"  # Verify join worked correctly
    assert result.iloc[0]["bqb"] == "BQB_IS"  # Verify bqb column is preserved

    # Test with multiple ontologies (set)
    result = match_by_ontology_and_identifier(
        feature_identifiers=feature_identifiers,
        species_identifiers=species_identifiers,
        ontologies={"chebi", "uniprot"}
    )
    assert len(result) == 2  # One chebi and one uniprot match
    assert set(result["ontology"]) == {"chebi", "uniprot"}  # From species_identifiers
    assert set(result["identifier"]) == {"15377", "P12345"}
    # Verify results are correctly matched
    chebi_row = result[result["ontology"] == "chebi"].iloc[0]
    uniprot_row = result[result["ontology"] == "uniprot"].iloc[0]
    assert chebi_row["results"] == 1.0
    assert uniprot_row["results"] == -1.0
    assert chebi_row["s_name"] == "compound1"
    assert uniprot_row["s_name"] == "protein1"
    assert chebi_row["bqb"] == "BQB_IS"
    assert uniprot_row["bqb"] == "BQB_IS"

    # Test with list of ontologies
    result = match_by_ontology_and_identifier(
        feature_identifiers=feature_identifiers,
        species_identifiers=species_identifiers,
        ontologies=["chebi", "uniprot"]
    )
    assert len(result) == 2
    assert set(result["ontology"]) == {"chebi", "uniprot"}  # From species_identifiers

    # Test with no matches
    no_match_features = pd.DataFrame({
        "ontology": ["chebi"],
        "identifier": ["99999"],
        "results": [1.0]
    })
    result = match_by_ontology_and_identifier(
        feature_identifiers=no_match_features,
        species_identifiers=species_identifiers,
        ontologies="chebi"
    )
    assert len(result) == 0

    # Test with empty features
    empty_features = pd.DataFrame({
        "ontology": [],
        "identifier": [],
        "results": []
    })
    result = match_by_ontology_and_identifier(
        feature_identifiers=empty_features,
        species_identifiers=species_identifiers,
        ontologies={"chebi", "uniprot"}
    )
    assert len(result) == 0

    # Test with invalid ontology
    with pytest.raises(ValueError, match="Invalid ontologies specified:.*"):
        match_by_ontology_and_identifier(
            feature_identifiers=feature_identifiers,
            species_identifiers=species_identifiers,
            ontologies="invalid_ontology"
        )

    # Test with ontology not in feature_identifiers
    result = match_by_ontology_and_identifier(
        feature_identifiers=feature_identifiers,
        species_identifiers=species_identifiers,
        ontologies={"ensembl_gene"}  # Only in species_identifiers
    )
    assert len(result) == 0

    # Test with custom feature_id_var
    feature_identifiers_custom = feature_identifiers.rename(
        columns={"identifier": "custom_id"}
    )
    result = match_by_ontology_and_identifier(
        feature_identifiers=feature_identifiers_custom,
        species_identifiers=species_identifiers,
        ontologies={"chebi"},
        feature_id_var="custom_id"
    )
    assert len(result) == 1
    assert result.iloc[0]["custom_id"] == "15377"
    assert result.iloc[0]["ontology"] == "chebi"  # From species_identifiers
    assert result.iloc[0]["s_name"] == "compound1"
    assert result.iloc[0]["bqb"] == "BQB_IS" 