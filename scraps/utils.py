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

def _ensure_nonnegative_vertex_attribute(g: ig.Graph, attribute: str):
    """
    Utility to check that a vertex attribute is present, numeric, and non-negative.
    Raises ValueError if checks fail.
    Missing or None values are treated as 0.
    Raises ValueError if attribute is missing for all vertices or all values are zero.
    """
    
    all_missing = all((attribute not in v.attributes() or v[attribute] is None) for v in g.vs)
    if all_missing:
        raise ValueError(f"Vertex attribute '{attribute}' is missing for all vertices.")
    
    values = [
        v[attribute] if (attribute in v.attributes() and v[attribute] is not None) else 0.0
        for v in g.vs
    ]
    
    arr = np.array(values, dtype=float)
    
    if np.all(arr == 0):
        raise ValueError(f"Vertex attribute '{attribute}' is zero for all vertices; cannot use as reset vector.")
    if np.any(arr < 0):
        raise ValueError(f"Attribute '{attribute}' contains negative values.")
    
    return arr


def personalized_pagerank_by_attribute(
    g: ig.Graph,
    attribute: str,
    damping: float = 0.85,
    calculate_uniform_dist: bool = True,
    additional_propagation_args: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Run personalized PageRank with reset probability proportional to a vertex attribute.
    Optionally computes uniform PPR over nonzero attribute nodes.

    Parameters
    ----------
    g : igraph.Graph
        The input graph.
    attribute : str
        The vertex attribute to use for personalization.
    damping : float, optional
        Damping factor (default 0.85).
    calculate_uniform_dist : bool, optional
        If True, also compute uniform PPR over nonzero attribute nodes.
    additional_propagation_args : dict, optional
        Additional arguments to pass to igraph's personalized_pagerank. Keys must match the method's signature.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['name', 'pagerank_by_attribute', attribute] and optionally 'pagerank_uniform'.

    Example
    -------
    >>> import igraph as ig
    >>> from scraps.utils import personalized_pagerank_by_attribute
    >>> g = ig.Graph.Full(3)
    >>> g.vs['name'] = ['A', 'B', 'C']
    >>> g.vs['score'] = [1, 0, 2]
    >>> df = personalized_pagerank_by_attribute(g, 'score')
    >>> print(df)
    """
    # Validate and extract attribute (missing/None as 0)
    attr = _ensure_nonnegative_vertex_attribute(g, attribute)

    # Validate additional_propagation_args
    if additional_propagation_args is None:
        additional_propagation_args = {}
    else:
        valid_args = set(inspect.signature(g.personalized_pagerank).parameters.keys())
        for k in additional_propagation_args:
            if k not in valid_args:
                raise ValueError(f"Invalid argument for personalized_pagerank: {k}")

    # Personalized PageRank (no normalization, igraph handles it)
    pr_attr = g.personalized_pagerank(reset=attr.tolist(), damping=damping, **additional_propagation_args)

    # Node names
    names = g.vs["name"] if "name" in g.vs.attributes() else list(range(g.vcount()))

    data = {
        "name": names,
        "pagerank_by_attribute": pr_attr,
        attribute: attr
    }

    # Uniform PPR over nonzero attribute nodes
    if calculate_uniform_dist:
        used_in_uniform = attr > 0
        n_uniform = used_in_uniform.sum()
        if n_uniform == 0:
            raise ValueError("No nonzero attribute values for uniform PPR.")
        uniform_vec = np.zeros_like(attr, dtype=float)
        uniform_vec[used_in_uniform] = 1.0 / n_uniform
        pr_uniform = g.personalized_pagerank(reset=uniform_vec.tolist(), damping=damping, **additional_propagation_args)
        data["pagerank_uniform"] = pr_uniform

    return pd.DataFrame(data)