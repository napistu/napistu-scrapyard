import pytest
import pandas as pd

from utils import _validate_wide_ontologies

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