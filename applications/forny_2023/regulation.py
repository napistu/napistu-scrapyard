import pandas as pd
import numpy as np
import igraph as ig
import inspect

from typing import Dict, Optional
from mudata import MuData

def split_varm_by_modality(mdata: MuData, varm_key: str = 'LFs') -> Dict[str, pd.DataFrame]:
    """
    Split a varm matrix by modality and join with corresponding var tables.
    
    Parameters
    ----------
    mdata : mudata.MuData
        MuData object containing multiple modalities and a varm matrix
    varm_key : str, optional
        Key in mdata.varm to split by modality. Default is 'LFs'
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with modality names as keys and DataFrames as values.
        Each DataFrame contains the var attributes and corresponding matrix values
        for that modality.
        
    Examples
    --------
    >>> # Split latent factors
    >>> modality_lfs = split_varm_by_modality(mdata, varm_key='LFs')
    >>> transcriptomics_lfs = modality_lfs['transcriptomics']
    >>> proteomics_lfs = modality_lfs['proteomics']
    >>>
    >>> # Split other varm matrix
    >>> modality_other = split_varm_by_modality(mdata, varm_key='other_matrix')
    """
    if varm_key not in mdata.varm:
        raise ValueError(f"No '{varm_key}' matrix found in varm")
    
    # Get the matrix and ensure it has the right index
    matrix = pd.DataFrame(
        mdata.varm[varm_key],
        index=mdata.var_names,
        columns=[f'{varm_key}{i+1}' for i in range(mdata.varm[varm_key].shape[1])]
    )
    
    # Initialize results dictionary
    results: Dict[str, pd.DataFrame] = {}
    
    # Process each modality
    for modality in mdata.mod.keys():
        # Get the var_names for this modality
        mod_vars = mdata.mod[modality].var_names
        
        # Extract matrix values for this modality
        mod_matrix = matrix.loc[mod_vars]
        
        # Get var table and ensure index matches
        mod_var = mdata.mod[modality].var.copy()
        
        # Verify index alignment
        if not mod_var.index.equals(mod_matrix.index):
            raise ValueError(
                f"Index mismatch in {modality}: var table and matrix subset have different indices"
            )
        
        # Join with var table on index
        mod_results = pd.concat([mod_var, mod_matrix], axis=1)
        
        # Store in results
        results[modality] = mod_results
    
    return results
