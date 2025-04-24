"""
This module contains functions for data normalization and transformation.
"""

from copy import deepcopy
from typing import Tuple, List, Dict, Union, Optional
import warnings

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

import matplotlib.pyplot as plt


def plot_feature_counts_histogram(
    adata: AnnData,
    min_counts: int = 10,
    layer: Optional[str] = None,
    log_scale: bool = True,
    n_bins: int = 50,
    figsize: Tuple[int, int] = (10, 6),
    save: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a histogram of total counts per feature across all samples with
    equally spaced bins on the transformed axis.
    
    Parameters
    ----------
    adata: anndata.AnnData
        The AnnData object to analyze
    min_counts: int, default=10
        Threshold to mark with a vertical line
    layer: str, optional
        If provided, use this layer instead of .X
    log_scale: bool, default=True
        Whether to use log scale for x-axis
    n_bins: int, default=50
        Number of bins for the histogram
    figsize: tuple, default=(10, 6)
        Figure size as (width, height) in inches
    save: str, optional
        Path to save the figure
    show: bool, default=True
        Whether to display the figure
    title: str, optional
        Custom title for the plot
    xlim: tuple, optional
        Custom x-axis limits as (min, max)
    
    Returns
    -------
    Tuple containing:
        - matplotlib.figure.Figure: The figure object
        - matplotlib.axes.Axes: The axes object
    """
    # Select the matrix to use (X or a specific layer)
    if layer is None:
        matrix = adata.X
        matrix_name = "X"
    else:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in AnnData object")
        matrix = adata.layers[layer]
        matrix_name = layer
    
    # Calculate the total counts per feature (gene/transcript)
    if sp.issparse(matrix):
        feature_counts = np.array(matrix.sum(axis=0)).flatten()
    else:
        feature_counts = np.sum(matrix, axis=0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle log scale with proper binning
    if log_scale:
        # Add a small pseudocount to avoid log(0)
        pseudocount = 1
        feature_counts_plus = feature_counts + pseudocount
        min_counts_plus = min_counts + pseudocount
        
        # Determine minimum non-zero value for log scale
        min_nonzero = np.min(feature_counts_plus[feature_counts_plus > 0])
        
        # Create log-spaced bins
        if xlim is not None:
            bin_min, bin_max = xlim
        else:
            bin_min = max(min_nonzero, 1)  # Ensure minimum is at least 1
            bin_max = np.max(feature_counts_plus) * 1.01  # Add 1% to include maximum
        
        bins = np.logspace(np.log10(bin_min), np.log10(bin_max), n_bins)
        
        # Plot histogram with log-spaced bins
        counts, bin_edges, _ = ax.hist(
            feature_counts_plus, 
            bins=bins, 
            alpha=0.7, 
            color='steelblue', 
            edgecolor='black'
        )
        
        # Set log scale for x-axis
        ax.set_xscale('log')
        x_label = "Total counts per feature (log scale)"
        
        # Add vertical line for min_counts threshold
        ax.axvline(x=min_counts_plus, color='red', linestyle='--', linewidth=2, 
                   label=f'Min counts threshold: {min_counts}')
    else:
        # For linear scale, use regular bins
        if xlim is not None:
            bin_min, bin_max = xlim
        else:
            bin_min = 0
            bin_max = np.max(feature_counts) * 1.01  # Add 1% to include maximum
        
        bins = np.linspace(bin_min, bin_max, n_bins)
        
        # Plot histogram with linear-spaced bins
        counts, bin_edges, _ = ax.hist(
            feature_counts, 
            bins=bins, 
            alpha=0.7, 
            color='steelblue', 
            edgecolor='black'
        )
        
        x_label = "Total counts per feature"
        
        # Add vertical line for min_counts threshold
        ax.axvline(x=min_counts, color='red', linestyle='--', linewidth=2, 
                   label=f'Min counts threshold: {min_counts}')
    
    # Count features below threshold
    n_below_threshold = np.sum(feature_counts < min_counts)
    percentage_below = (n_below_threshold / len(feature_counts)) * 100
    
    # Add annotations
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of features")
    
    if title is None:
        title = f"Distribution of total counts per feature ({matrix_name})"
    title += f"\n{n_below_threshold} features ({percentage_below:.1f}%) have < {min_counts} counts"
    ax.set_title(title)
    
    # Set x-axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save is not None:
        fig.savefig(save, bbox_inches='tight', dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    
    return fig, ax


def filter_features_by_counts(
    adata: AnnData,
    min_counts: int = 10,
    layer: Optional[str] = None,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Filter features to retain only those with more than min_counts reads across all samples.
    
    Parameters
    ----------
    adata: anndata.AnnData
        The AnnData object to filter
    min_counts: int, default=10
        Minimum number of counts required for a feature to be kept
    layer: str, optional
        If provided, use this layer instead of .X for filtering
    inplace: bool, default=True
        Whether to modify the AnnData object in place or return a copy
    
    Returns
    -------
    anndata.AnnData, optional
        Filtered AnnData object (only if inplace=False)
    """
    # Work with a copy if not inplace
    if not inplace:
        adata = adata.copy()
    
    # Select the matrix to use (X or a specific layer)
    if layer is None:
        matrix = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in AnnData object")
        matrix = adata.layers[layer]
    
    # Calculate the total counts per feature (gene/transcript)
    if sp.issparse(matrix):
        feature_counts = np.array(matrix.sum(axis=0)).flatten()
    else:
        feature_counts = np.sum(matrix, axis=0)
    
    # Create a mask for features with counts > min_counts
    keep_features = feature_counts > min_counts
    
    # Apply the filter
    if inplace:
        adata._inplace_subset_var(keep_features)
        filtered_adata = adata
    else:
        filtered_adata = adata[:, keep_features]
    
    # Store filtering information
    n_features_before = adata.n_vars if not inplace else filtered_adata.n_vars + np.sum(~keep_features)
    n_features_after = filtered_adata.n_vars
    filtered_adata.uns["feature_filtering"] = {
        "min_counts": min_counts,
        "n_features_before": n_features_before,
        "n_features_after": n_features_after,
        "n_features_removed": n_features_before - n_features_after,
    }
    
    if not inplace:
        return filtered_adata


def log2_transform(
    adata: AnnData,
    layer: Optional[str] = None,
    inplace: bool = True,
    pseudocount: float = 1.0,
    create_layer: bool = True,
    new_layer_name: Optional[str] = None,
    modify_source: bool = False
) -> Optional[AnnData]:
    """
    Log2-transforms the data matrix of an AnnData object.
    
    Parameters
    ----------
    adata: AnnData
        The AnnData object to transform
    layer: str, optional
        If provided, transforms this layer instead of .X
    inplace: bool, default=True
        Whether to modify the AnnData object in place or return a copy
    pseudocount: float, default=1.0
        Value to add to the data before log-transforming to avoid log(0)
    create_layer: bool, default=True
        Whether to store the transformed data in a new layer
    new_layer_name: str, optional
        Name for the new layer containing transformed data.
        If None, defaults to "log2" or "{layer}_log2"
    modify_source: bool, default=False
        Whether to modify the source data matrix. If False, the original
        data will be preserved even when create_layer=True.
    
    Returns
    -------
    Optional[AnnData]
        The log2-transformed object (only if inplace=False)
    """
    # Work with a copy if not inplace
    if not inplace:
        adata = adata.copy()
    
    # Function to transform a matrix
    def transform_matrix(X: Union[np.ndarray, sp.spmatrix]) -> Union[np.ndarray, sp.spmatrix]:
        """Transform a single data matrix using log2."""
        if sp.issparse(X):
            X_transformed = X.copy()
            X_transformed.data = np.log2(X_transformed.data + pseudocount)
            return X_transformed
        else:
            return np.log2(X + pseudocount)
    
    # Determine the source matrix to transform
    if layer is None:
        source_matrix = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in AnnData object")
        source_matrix = adata.layers[layer]
    
    # Determine the name for the new layer if creating one
    if new_layer_name is None:
        if layer is None:
            new_layer_name = "log2"
        else:
            new_layer_name = f"{layer}_log2"
    
    # Create a new layer with transformed data if requested
    if create_layer:
        adata.layers[new_layer_name] = transform_matrix(source_matrix)
    
    # Only modify the source matrix if explicitly requested
    if modify_source:
        if layer is None:
            if sp.issparse(adata.X):
                adata.X.data = np.log2(adata.X.data + pseudocount)
            else:
                adata.X = np.log2(adata.X + pseudocount)
        else:
            if sp.issparse(adata.layers[layer]):
                adata.layers[layer].data = np.log2(adata.layers[layer].data + pseudocount)
            else:
                adata.layers[layer] = np.log2(adata.layers[layer] + pseudocount)
    
    # Add a note in the var attributes to track the transformation
    if create_layer:
        adata.uns[f"{new_layer_name}_transform"] = {
            "type": "log2",
            "pseudocount": pseudocount,
            "source": "X" if layer is None else layer
        }
    
    if not inplace:
        return adata


def center_rows_and_columns(
    adata: AnnData,
    layer: Optional[str] = None,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
    inplace: bool = True,
    copy_layer: bool = False,
    new_layer_name: Optional[str] = None,
) -> Optional[AnnData]:
    """
    Alternately center rows (features) and columns (cells) of an AnnData object
    until both are approximately zero-centered.
    
    Parameters
    ----------
    adata: anndata.AnnData
        The AnnData object to transform
    layer: str, optional
        If provided, transforms this layer instead of .X
    max_iterations: int, default=10
        Maximum number of iterations to perform
    tolerance: float, default=1e-6
        Convergence threshold for mean absolute deviation
    inplace: bool, default=True
        Whether to modify the AnnData object in place or return a copy
    copy_layer: bool, default=False
        Whether to create a new layer with the result
    new_layer_name: str, optional
        Name for the new layer containing transformed data.
        If None but copy_layer is True, defaults to "centered" or "{layer}_centered"
    
    Returns
    -------
    anndata.AnnData, optional
        The centered object (only if inplace=False)
    """
    # Work with a copy if not inplace
    if not inplace:
        adata = adata.copy()
    
    # Determine the source matrix and where to store the result
    if layer is None:
        source_matrix = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in AnnData object")
        source_matrix = adata.layers[layer]
    
    # Make a copy of the data matrix
    if sp.issparse(source_matrix):
        # For sparse matrices, we need to convert to dense as centering will likely
        # destroy sparsity and lead to inefficient computations
        matrix = source_matrix.toarray()
    else:
        matrix = source_matrix.copy()
    
    # Initialize variables for tracking convergence
    n_rows, n_cols = matrix.shape
    converged = False
    
    # Alternate centering
    for i in range(max_iterations):
        # Center rows (features)
        row_means = np.mean(matrix, axis=1, keepdims=True)
        matrix = matrix - row_means
        
        # Center columns (cells)
        col_means = np.mean(matrix, axis=0, keepdims=True)
        matrix = matrix - col_means
        
        # Check convergence
        row_mean_abs = np.mean(np.abs(np.mean(matrix, axis=1)))
        col_mean_abs = np.mean(np.abs(np.mean(matrix, axis=0)))
        
        if row_mean_abs < tolerance and col_mean_abs < tolerance:
            converged = True
            break
    
    # Warn if not converged
    if not converged:
        warnings.warn(
            f"Centering did not converge after {max_iterations} iterations. "
            f"Final mean absolute deviations: rows={row_mean_abs:.2e}, columns={col_mean_abs:.2e}. "
            f"Consider increasing max_iterations or adjusting tolerance."
        )
    
    # Store the result
    if layer is None:
        adata.X = matrix
    else:
        if copy_layer:
            # Determine new layer name
            if new_layer_name is None:
                new_layer_name = f"{layer}_centered" if layer else "centered"
            adata.layers[new_layer_name] = matrix
        else:
            adata.layers[layer] = matrix
    
    if not inplace:
        return adata


def center_rows_and_columns_mudata(
    mdata,  # Not type-hinting to avoid import error for MuData
    modalities: Optional[Union[str, List[str]]] = None,
    layer: Optional[Union[str, Dict[str, str]]] = None,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
    inplace: bool = True,
    copy_layer: bool = True,  # Changed default to True
    new_layer_name: Optional[Union[str, Dict[str, str]]] = None,
) -> Optional:
    """
    Apply alternate row-column centering to specified modalities in a MuData object.
    """
    # Work with a copy if not inplace
    if not inplace:
        mdata = deepcopy(mdata)
    
    # Determine which modalities to transform
    if modalities is None:
        modalities = list(mdata.mod.keys())
    elif isinstance(modalities, str):
        modalities = [modalities]
    
    # Handle layer parameter
    if isinstance(layer, str) or layer is None:
        layer_dict = {mod: layer for mod in modalities}
    else:
        layer_dict = layer
    
    # Handle new_layer_name parameter
    if isinstance(new_layer_name, str) or new_layer_name is None:
        new_layer_dict = {mod: new_layer_name for mod in modalities}
    else:
        new_layer_dict = new_layer_name
    
    # Apply centering to each modality
    for mod in modalities:
        if mod not in mdata.mod:
            raise ValueError(f"Modality '{mod}' not found in MuData object")
        
        mod_layer = layer_dict.get(mod)
        mod_new_layer = new_layer_dict.get(mod) if new_layer_dict else None
        
        center_rows_and_columns(
            mdata.mod[mod],
            layer=mod_layer,
            max_iterations=max_iterations,
            tolerance=tolerance,
            inplace=True,  # Always inplace for the individual modality
            copy_layer=copy_layer,  # This should be True to create new layers
            new_layer_name=mod_new_layer
        )
    
    if not inplace:
        return mdata