import os
from typing import List, Optional, Tuple, Dict, Union

from anndata import AnnData
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from mudata import MuData
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr, pearsonr
import seaborn as sns


# Set up the plotting style with a standard matplotlib style
plt.style.use("default")
sc.settings.set_figure_params(dpi=100, frameon=True)


def plot_mudata_pca_variance(
    mdata: MuData,
    modalities: Optional[List[str]] = None,
    n_pcs: int = 10,
    pca_kwargs: Optional[Dict] = None,
    plot_grid: Optional[Tuple[int, int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save: Optional[str] = None
) -> Dict:
    """
    Create side-by-side PCA variance plots for multiple modalities in a MuData object.
    
    Parameters
    ----------
    mdata : MuData
        Multi-modal data object containing multiple AnnData objects
    modalities : list, optional
        List of modality names to analyze. If None, uses all modalities in mdata
    n_pcs : int, optional
        Number of principal components to include (default: 10)
    pca_kwargs : dict, optional
        Additional arguments to pass to sc.pp.pca()
    plot_grid : tuple, optional
        Grid dimensions (rows, cols) for subplot arrangement. If None, automatically determined.
    figsize : tuple, optional
        Figure size (width, height) in inches. If None, calculated based on number of modalities.
    save : str, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns
    -------
    dict
        Dictionary containing all generated data and figures for each modality
    """
    # Set defaults
    if modalities is None:
        modalities = list(mdata.mod.keys())
    
    if pca_kwargs is None:
        pca_kwargs = {}
    
    # Determine grid layout for plots
    if plot_grid is None:
        if len(modalities) <= 3:
            plot_grid = (1, len(modalities))
        else:
            # Arrange in a square-ish grid
            n_cols = int(np.ceil(np.sqrt(len(modalities))))
            n_rows = int(np.ceil(len(modalities) / n_cols))
            plot_grid = (n_rows, n_cols)
    
    # Calculate figsize if not provided
    if figsize is None:
        # Base size per plot
        base_width, base_height = 5, 4
        figsize = (base_width * plot_grid[1], base_height * plot_grid[0])
    
    # Create figure with subplots
    fig, axes = plt.subplots(plot_grid[0], plot_grid[1], figsize=figsize)
    
    # Flatten axes array for easier indexing
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Dictionary to store results
    results = {}
    
    # Generate colors from a colormap based on the number of modalities
    cmap = plt.cm.tab10  # Use tab10 colormap for distinct colors
    colors = [cmap(i % 10) for i in range(len(modalities))]  # Cycle through 10 colors
    
    # Process each modality
    for idx, modality in enumerate(modalities):
        if modality not in mdata.mod:
            raise ValueError(f"Modality '{modality}' not found in MuData object")
        
        # Create entry in results dictionary
        results[modality] = {}
        
        # Get AnnData for this modality
        adata = mdata.mod[modality]
        
        # Compute PCA if needed
        _compute_pca_if_needed(adata, **pca_kwargs)
        
        # Select color palette for this modality
        bar_color = colors[idx]
        line_color = 'black'  # Use black for cumulative line
        threshold_color = 'red'  # Use red for threshold
        
        color_palette = {
            'bar': bar_color,
            'line': line_color,
            'threshold': threshold_color
        }
        
        # Visualize correlations if we have room
        if idx < len(axes):
            ax = axes[idx]
            subfig, plot_df = _plot_pca_variance(
                adata, n_pcs, ax, modality, color_palette
            )
            results[modality]['plot_df'] = plot_df
        else:
            print(f"Warning: Not enough subplots for modality '{modality}'")
    
    # Hide any unused subplots
    for i in range(len(modalities), len(axes)):
        axes[i].set_visible(False)
    
    # Set title for the whole figure
    fig.suptitle("PCA Variance by Modality", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    
    # Save the figure if requested
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    
    # Store the figure in results
    results['figure'] = fig
    
    return results


def analyze_pc_metadata_correlation_mudata(
    mdata: MuData,
    modalities: Optional[List[str]] = None,
    n_pcs: int = 10,
    metadata_vars: Optional[List[str]] = None,
    prioritized_vars: Optional[List[str]] = None,
    corr_method: str = "spearman",
    cmap: str = "RdBu_r",
    size_scale: float = 50,
    figsize: Optional[Tuple[float, float]] = None,
    pval_threshold: float = 0.05,
    obsm_key: str = "X_pca",
    pca_kwargs: Optional[Dict] = None,
    plot_grid: Optional[Tuple[int, int]] = None,
    save: Optional[str] = None,
    max_metadata_vars: int = 20,
    max_label_length: int = 25,
    min_values_threshold: float = 0.5  # Minimum fraction of non-missing values required
) -> Dict:
    """
    Complete workflow for analyzing correlations between metadata and principal components
    across modalities in a MuData object.
    
    Parameters
    ----------
    mdata : MuData
        Multi-modal data object containing multiple AnnData objects
    modalities : list, optional
        List of modality names to analyze. If None, uses all modalities in mdata
    n_pcs : int, optional
        Number of principal components to include (default: 10)
    metadata_vars : list, optional
        List of metadata variables to correlate. If None, uses all categorical and numeric columns.
    prioritized_vars : list, optional
        List of metadata variables to always include at the top of the plot, regardless of correlation
    corr_method : str, optional
        Correlation method: 'spearman' or 'pearson' (default: 'spearman')
    cmap : str, optional
        Colormap for correlation values (default: 'RdBu_r')
    size_scale : float, optional
        Scaling factor for dot sizes (default: 50)
    figsize : tuple, optional
        Figure size (width, height) in inches. If None, calculated based on number of modalities.
    pval_threshold : float, optional
        Threshold for statistical significance (default: 0.05)
    obsm_key : str, optional
        Key in adata.obsm containing PCA coordinates (default: 'X_pca')
    pca_kwargs : dict, optional
        Additional arguments to pass to sc.pp.pca()
    plot_grid : tuple, optional
        Grid dimensions (rows, cols) for subplot arrangement. If None, automatically determined.
    save : str, optional
        Path to save the figure. If None, the figure is not saved.
    max_metadata_vars : int, optional
        Maximum number of metadata variables to show in plots (default: 20)
    max_label_length : int, optional
        Maximum length for metadata variable names before truncation (default: 25)
    min_values_threshold : float, optional
        Minimum fraction of non-missing values required (default: 0.5)
        
    Returns
    -------
    dict
        Dictionary containing all generated data and figures for each modality
    """
    # Initialize parameters and setup
    modalities, plot_grid, pca_kwargs, prioritized_vars = _initialize_parameters(
        mdata, modalities, plot_grid, pca_kwargs, prioritized_vars)
    
    # Set up figure and grid
    fig, gs, figsize = _setup_figure_and_grid(
        modalities, plot_grid, figsize)
    
    # Process all modalities to collect correlation data
    all_correlation_data, all_pvalue_data, all_metadata_vars, results = _process_all_modalities(
        mdata, modalities, obsm_key, n_pcs, metadata_vars, pca_kwargs, corr_method, 
        min_values_threshold)  # Add min_values_threshold
    
    # Determine shared variable order across modalities
    shared_var_order, truncated_names = _determine_shared_variable_order(
        all_metadata_vars, all_correlation_data, prioritized_vars, 
        max_metadata_vars, max_label_length, modalities)
    
    # Create plots for each modality
    _create_plots_for_each_modality(
        modalities, all_correlation_data, all_pvalue_data, shared_var_order,
        truncated_names, plot_grid, gs, corr_method, cmap, size_scale,
        pval_threshold, results)
    
    # Add colorbar and finalize figure
    _add_colorbar_and_finalize(fig, gs, cmap, corr_method, save)
    
    results['figure'] = fig
    return results


def plot_pcs(adata: AnnData, 
           pc_x: int = 1, 
           pc_y: int = 2, 
           metadata_attrs: Optional[Union[str, List[str]]] = None, 
           color_palette: Optional[Union[Dict, List]] = None, 
           figsize: Optional[Tuple[float, float]] = None, 
           n_cols: Optional[int] = None, 
           legend: bool = True, 
           save: Optional[str] = None, 
           point_size: int = 30,
           pca_kwargs: Optional[Dict] = None):
    """
    Create PC plots for an AnnData object, with one subplot for each metadata attribute.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to plot
    pc_x : int, default=1
        X-axis principal component (1-based indexing, so 1 means PC1)
    pc_y : int, default=2
        Y-axis principal component (1-based indexing, so 2 means PC2)
    metadata_attrs : list of str or str, optional
        List of metadata attributes from adata.obs to use for coloring points.
        If None, will use a single plot with no color coding.
    color_palette : dict or list, optional
        Color mapping for values in metadata attributes. If None, default colors will be used.
    figsize : tuple, optional
        Figure size (width, height) in inches. If None, calculated based on number of plots.
    n_cols : int, optional
        Number of columns in the grid layout. If None, will be calculated automatically.
    legend : bool, default=True
        Whether to show a legend.
    save : str, optional
        Path to save the figure. If None, the figure is not saved.
    point_size : int, default=30
        Size of points in the scatter plot
    pca_kwargs : dict, optional
        Additional arguments to pass to sc.pp.pca() if PCA needs to be computed
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing all plots
    """
    # Set default pca_kwargs if not provided
    if pca_kwargs is None:
        pca_kwargs = {}
    
    # Compute PCA if needed
    _compute_pca_if_needed(adata, **pca_kwargs)
    
    # Get PCA coordinates - adjust for 0-based indexing
    pc_idx_x = pc_x - 1  # Convert from 1-based to 0-based
    pc_idx_y = pc_y - 1  # Convert from 1-based to 0-based
    
    # Check if indices are valid
    if pc_idx_x < 0 or pc_idx_y < 0:
        raise ValueError("PC indices must be positive integers (1-based indexing)")
    
    # Check if requested PCs are available
    max_pc = adata.obsm['X_pca'].shape[1]
    if pc_idx_x >= max_pc or pc_idx_y >= max_pc:
        raise ValueError(f"Requested PCs exceed available components (max: {max_pc})")
    
    # Get coordinates
    pcx = adata.obsm['X_pca'][:, pc_idx_x]
    pcy = adata.obsm['X_pca'][:, pc_idx_y]
    
    # PC names for labels
    pc_x_name = f"PC{pc_x}"
    pc_y_name = f"PC{pc_y}"
    
    # Set up metadata attributes
    if metadata_attrs is None:
        metadata_attrs = [None]  # Single plot with no coloring
    elif isinstance(metadata_attrs, str):
        # Handle single string input
        metadata_attrs = [metadata_attrs]
    
    # Verify all metadata attributes exist in the dataset
    for attr in metadata_attrs:
        if attr is not None and attr not in adata.obs.columns:
            raise ValueError(f"Metadata attribute '{attr}' not found in adata.obs")
    
    # Calculate layout
    n_plots = len(metadata_attrs)
    if n_cols is None:
        n_cols = min(3, n_plots)  # Default: max 3 plots per row
    n_rows = int(np.ceil(n_plots / n_cols))
    
    # Adjust figsize based on number of plots
    if figsize is None:
        # Base size for each plot
        width_per_plot = 4.5
        height_per_plot = 4.5
        
        # Add extra space for legends if needed
        if legend:
            # More height for multiple rows with legends
            if n_rows > 1:
                fig_height = height_per_plot * n_rows + 2  # Extra space for legends
            else:
                fig_height = height_per_plot * n_rows + 1  # Less extra space for single row
        else:
            # Without legends, we need less space
            fig_height = height_per_plot * n_rows + 0.5
            
        fig_width = width_per_plot * n_cols
        figsize = (fig_width, fig_height)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                           squeeze=False)
    
    # Set spacing based on legend status
    if n_rows > 1:
        if legend:
            plt.subplots_adjust(wspace=0.25, hspace=0.5, top=0.9, bottom=0.1)
        else:
            plt.subplots_adjust(wspace=0.25, hspace=0.3, top=0.9, bottom=0.1)
    else:
        if legend:
            plt.subplots_adjust(wspace=0.25, hspace=0.3, top=0.85, bottom=0.2)
        else:
            plt.subplots_adjust(wspace=0.25, hspace=0.3, top=0.85, bottom=0.15)
    
    # Default color if no metadata attribute is provided
    default_color = "#1f77b4"  # Standard matplotlib blue
    
    # Calculate global data ranges for consistent scaling
    x_min, x_max = np.nanmin(pcx), np.nanmax(pcx)
    y_min, y_max = np.nanmin(pcy), np.nanmax(pcy)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range) * 1.1  # 10% padding
    
    # Set limits centered on the data
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # Process each metadata attribute
    for i, attr in enumerate(metadata_attrs):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        # Apply consistent axis limits
        ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
        
        # Force square aspect ratio
        ax.set_aspect('equal', 'box')
        
        if attr is not None:
            # Check if attribute is numeric with more than 2 levels
            is_numeric_multilevel = _is_numeric_attribute(adata.obs[attr])
            
            if is_numeric_multilevel:
                # Use colorbar for numeric attributes with more than 2 values
                # Get non-NaN values
                non_nan_mask = ~adata.obs[attr].isna()
                values = adata.obs[attr][non_nan_mask]
                
                # Create colormap
                cmap = plt.colormaps['viridis']
                norm = Normalize(vmin=values.min(), vmax=values.max())
                
                # Plot non-NaN points with colormap
                scatter = ax.scatter(
                    pcx[non_nan_mask],
                    pcy[non_nan_mask],
                    c=values,
                    cmap=cmap,
                    norm=norm,
                    alpha=0.7,
                    s=point_size,
                    edgecolor='w',
                    linewidth=0.2
                )
                
                # Plot NaN points separately in gray
                nan_mask = adata.obs[attr].isna()
                if np.any(nan_mask):
                    ax.scatter(
                        pcx[nan_mask],
                        pcy[nan_mask],
                        c='#cccccc',  # Light gray
                        alpha=0.7,
                        s=point_size,
                        edgecolor='w',
                        linewidth=0.2
                    )
                
                # Add a separate axis for colorbar to avoid size distortion
                cax_pos = ax.get_position()
                cax = fig.add_axes([
                    cax_pos.x0 + cax_pos.width * 0.1,  # Centered horizontally 
                    cax_pos.y0 - 0.1,                  # Below the plot
                    cax_pos.width * 0.8,               # 80% of plot width
                    0.02                               # Fixed height
                ])
                
                # Add colorbar with separate axis
                cbar = fig.colorbar(
                    scatter,
                    cax=cax,
                    orientation='horizontal'
                )
                
                # Remove colorbar label (redundant with title)
            else:
                # For categorical or binary attributes, use discrete colors
                # Get unique values directly, keeping NaN values
                categories = pd.Series(adata.obs[attr].unique()).sort_values(na_position='last')
                
                # Create color mapping with matplotlib colors
                cmap = plt.colormaps['tab10']
                category_colors = {j: cmap(j % 10) for j in range(len(categories))}
                
                # Special color for NaN values (gray)
                nan_color = "#cccccc"  # Light gray
                
                # Create scatter plots for each category
                for j, cat in enumerate(categories):
                    # Create mask
                    mask = _create_category_mask(adata.obs, attr, cat)
                    
                    if np.sum(mask) > 0:  # Only plot if there are data points
                        # Use special color for NaN values
                        if pd.isna(cat):
                            color = nan_color
                        else:
                            color = category_colors[j]
                        
                        ax.scatter(
                            pcx[mask], 
                            pcy[mask],
                            c=[color],
                            alpha=0.7,
                            s=point_size,
                            edgecolor='w',
                            linewidth=0.2
                        )
                
                # Create formatted labels for legend
                legend_labels = [_format_value_for_legend(cat) for cat in categories]
                
                # Create legend
                if legend:
                    # Determine number of columns based on number of categories
                    num_cols = min(3, len(categories))
                    
                    handles = []
                    for j, (cat, label) in enumerate(zip(categories, legend_labels)):
                        if pd.isna(cat):
                            color = nan_color
                        else:
                            color = category_colors[j]
                        
                        handles.append(
                            Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color, 
                                  label=label, 
                                  markersize=8)
                        )
                    
                    # Place legend below plot
                    ax.legend(handles=handles, 
                              loc='upper center', 
                              bbox_to_anchor=(0.5, -0.15),
                              ncol=num_cols)
            
            # Set title
            ax.set_title(f"{attr}", pad=5)
        else:
            # No attribute provided, use default color
            ax.scatter(
                pcx, 
                pcy,
                c=default_color,
                alpha=0.7,
                s=point_size,
                edgecolor='w',
                linewidth=0.2
            )
            ax.set_title(f"{pc_x_name} vs {pc_y_name}", pad=5)
        
        # Set labels
        ax.set_xlabel(pc_x_name, labelpad=5)
        ax.set_ylabel(pc_y_name, labelpad=5)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Hide unused subplots
    for i in range(n_plots, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    # Add overall title
    if n_plots > 1:
        fig.suptitle(f'{pc_x_name} vs {pc_y_name} plots colored by sample attributes', 
                    fontsize=16, y=0.95, verticalalignment='top')
    
    # Save figure if requested
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    return fig


def plot_mudata_pcs(mdata: MuData, 
                   pc_x: int = 1, 
                   pc_y: int = 2, 
                   modalities: Optional[List[str]] = None, 
                   metadata_attrs: Optional[Union[str, List[str]]] = None, 
                   figsize: Optional[Tuple[float, float]] = None, 
                   legend: bool = True, 
                   save: Optional[str] = None, 
                   point_size: int = 30,
                   pca_kwargs: Optional[Dict] = None):
    """
    Create PC plots for multiple modalities in a MuData object.
    
    Parameters
    ----------
    mdata : MuData
        Multi-modal data object containing multiple AnnData objects
    pc_x : int, default=1
        X-axis principal component (1-based indexing, so 1 means PC1)
    pc_y : int, default=2
        Y-axis principal component (1-based indexing, so 2 means PC2)
    modalities : list of str, optional
        List of modality names to plot. If None, uses all modalities in mdata.
    metadata_attrs : list of str or str, optional
        List of metadata attributes from mdata.obs to use for coloring points.
        If None, will create one plot per modality with default coloring.
    figsize : tuple, optional
        Figure size (width, height) in inches. If None, calculated based on number of plots.
    legend : bool, default=True
        Whether to show a legend.
    save : str, optional
        Path to save the figure. If None, the figure is not saved.
    point_size : int, default=30
        Size of points in the scatter plot
    pca_kwargs : dict, optional
        Additional arguments to pass to sc.pp.pca() if PCA needs to be computed
        
    Returns
    -------
    dict
        Dictionary of figures, with modality names as keys
    """
    # Set default pca_kwargs if not provided
    if pca_kwargs is None:
        pca_kwargs = {}
        
    # Set modalities if not provided
    if modalities is None:
        modalities = list(mdata.mod.keys())
    
    # Create a separate figure for each modality
    figures = {}
    
    for modality in modalities:
        # Get the AnnData object for this modality
        adata = mdata.mod[modality]
        
        # Adjust the figure title
        print(f"Processing {modality} modality...")
        
        # Handle save path
        if save:
            # Create a modality-specific save path if needed
            mod_save = save
            if len(modalities) > 1:
                # If multiple modalities, add modality name to save path
                base, ext = os.path.splitext(save)
                mod_save = f"{base}_{modality}{ext}"
        else:
            mod_save = None
        
        # Generate the plot for this modality
        fig = plot_pcs(
            adata=adata,
            pc_x=pc_x,
            pc_y=pc_y,
            metadata_attrs=metadata_attrs,
            color_palette=None,
            figsize=figsize,
            legend=legend,
            save=mod_save,
            point_size=point_size,
            pca_kwargs=pca_kwargs
        )
        
        # Update the title to include modality
        if fig:
            pc_x_name = f"PC{pc_x}"
            pc_y_name = f"PC{pc_y}"
            # Set title with modality name
            fig.suptitle(f"{modality.capitalize()} {pc_x_name} vs {pc_y_name} plots colored by sample attributes",
                      fontsize=16, y=0.95, verticalalignment='top')
            figures[modality] = fig
    
    return figures


def _compute_pca_if_needed(adata: AnnData, **pca_kwargs) -> AnnData:
    """
    Compute PCA if not already present in the AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    **pca_kwargs
        Additional arguments to pass to sc.pp.pca()
    
    Returns
    -------
    AnnData
        The input AnnData with PCA computed
    """
    # Check if PCA is already computed
    if 'X_pca' not in adata.obsm.keys():
        sc.pp.pca(adata, **pca_kwargs)
    return adata


def _plot_pca_variance(
    adata: AnnData,
    n_pcs: int = 10,
    ax: Optional[plt.Axes] = None,
    modality: Optional[str] = None,
    color_palette: Optional[Dict[str, str]] = None
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Plot PCA variance for a single modality.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with PCA computed
    n_pcs : int, optional
        Number of principal components to show (default: 10)
    ax : matplotlib.Axes, optional
        Axes to plot on, new axes created if None
    modality : str, optional
        Name of the modality (for plot title)
    color_palette : dict, optional
        Dictionary with keys 'bar', 'line', 'threshold' for colors
        
    Returns
    -------
    tuple
        (fig, plot_df) containing the figure and plotting data
    """
    # Set default colors if not provided
    if color_palette is None:
        color_palette = {
            'bar': '#1f77b4',
            'line': '#ff7f0e',
            'threshold': '#ff7f0e'
        }
    
    # Get variance ratio data
    var_ratio = adata.uns['pca']['variance_ratio']
    
    # Limit to n_pcs
    n_pcs = min(n_pcs, len(var_ratio))
    var_ratio = var_ratio[:n_pcs]
    
    # Create new axis if not provided
    create_new_fig = ax is None
    if create_new_fig:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    
    # Plot variance explained (bar plot)
    ax.bar(np.arange(1, n_pcs+1), var_ratio, color=color_palette['bar'], 
           alpha=0.7, label='Variance explained')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Ratio')
    
    if modality:
        ax.set_title(f"{modality.capitalize()}")
    
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks to be integers
    ax.set_xticks(np.arange(1, n_pcs+1, max(1, n_pcs//10)))
    
    # Show cumulative variance
    ax2 = ax.twinx()
    cum_var = np.cumsum(var_ratio)
    ax2.plot(np.arange(1, n_pcs+1), cum_var, 
            color=color_palette['line'], marker='o', markersize=4,
            label='Cumulative variance')
    ax2.set_ylabel('Cumulative Variance Ratio', color=color_palette['line'])
    ax2.tick_params(axis='y', labelcolor=color_palette['line'])
    ax2.set_ylim([0, min(1.05, max(cum_var) * 1.1)])
    
    # Add horizontal line at 0.8 cumulative variance
    cum_var_threshold = 0.8
    ax2.axhline(y=cum_var_threshold, color=color_palette['threshold'], 
               linestyle='--', alpha=0.5,
               label=f'{cum_var_threshold:.0%} threshold')
    
    # Find where cumulative variance exceeds threshold
    threshold_idx = np.argmax(cum_var >= cum_var_threshold)
    if threshold_idx < len(cum_var):
        threshold_pc = threshold_idx + 1
        ax2.annotate(f'PC{threshold_pc}', 
                    xy=(threshold_pc, cum_var_threshold),
                    xytext=(threshold_pc + 1, cum_var_threshold - 0.1),
                    arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-0.2"),
                    horizontalalignment='left',
                    verticalalignment='top')
    
    # Create DataFrame with results
    plot_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(n_pcs)],
        'Variance': var_ratio,
        'Cumulative': cum_var
    })
    
    return fig, plot_df


# corrplot utils

def _initialize_parameters(mdata, modalities, plot_grid, pca_kwargs, prioritized_vars):
    """Initialize and validate input parameters."""
    if modalities is None:
        modalities = list(mdata.mod.keys())
    
    if pca_kwargs is None:
        pca_kwargs = {}
    
    if prioritized_vars is None:
        prioritized_vars = []
    
    # Determine grid layout for plots
    if plot_grid is None:
        if len(modalities) <= 3:
            plot_grid = (1, len(modalities))
        else:
            # Arrange in a square-ish grid
            n_cols = int(np.ceil(np.sqrt(len(modalities))))
            n_rows = int(np.ceil(len(modalities) / n_cols))
            plot_grid = (n_rows, n_cols)
    
    return modalities, plot_grid, pca_kwargs, prioritized_vars


def _setup_figure_and_grid(modalities, plot_grid, figsize):
    """Set up the figure and grid for plotting."""
    # Calculate figsize if not provided
    if figsize is None:
        # Base size per plot with extra space
        base_width, base_height = 7, 6  # Increased size
        # Add extra width for left plot with y-axis labels, less for others
        width_adjustment = 1.2 if len(modalities) > 1 else 0
        figsize = (base_width * plot_grid[1] + width_adjustment, base_height * plot_grid[0] + 2.0)
    
    # Create figure with more top space
    fig = plt.figure(figsize=figsize)
    
    # Define margins
    top_margin = 0.85  # For main title
    bottom_margin = 0.1
    
    # Define column width ratios
    if len(modalities) > 1:
        width_ratios = [1.3] + [0.8] * (plot_grid[1] - 1) + [0.15]  # First column wider, last for colorbar
    else:
        width_ratios = [1] + [0.15]  # Single plot plus colorbar
    
    # Create grid specification
    gs = plt.GridSpec(
        plot_grid[0], 
        plot_grid[1] + 1,  # Add an extra column for colorbar
        figure=fig,
        hspace=0.4,  # Extra vertical space between plots
        wspace=0.1,  # Reduced space between plots since we're removing most y-axes
        width_ratios=width_ratios,  # Control the width of each column
        top=top_margin,  # Added top margin for main title
        bottom=bottom_margin  # Bottom margin
    )
    
    return fig, gs, figsize


def _process_all_modalities(mdata, modalities, obsm_key, n_pcs, metadata_vars, 
                           pca_kwargs, corr_method, min_values_threshold=0.5):
    """Process all modalities to collect correlation data."""
    all_correlation_data = {}
    all_pvalue_data = {}
    all_metadata_vars = set()
    results = {}
    
    for modality in modalities:
        if modality not in mdata.mod:
            raise ValueError(f"Modality '{modality}' not found in MuData object")
        
        # Create entry in results dictionary
        results[modality] = {}
        
        # Get AnnData for this modality
        adata = mdata.mod[modality]
        
        # Compute PCA if needed
        _compute_pca_if_needed(adata, **pca_kwargs)
        
        # Preprocess the data
        pc_df, metadata_df = _preprocess_metadata_for_correlation(
            adata, metadata_vars, obsm_key, n_pcs, min_values_threshold
        )
        results[modality]['pc_df'] = pc_df
        results[modality]['metadata_df'] = metadata_df
        
        # Skip if no metadata variables found
        if metadata_df.shape[1] == 0:
            print(f"No suitable metadata variables found for modality '{modality}'")
            continue
        
        # Calculate correlations
        corr_df, pval_df = _calculate_pc_metadata_correlations(
            pc_df, metadata_df, corr_method
        )
        results[modality]['corr_df'] = corr_df
        results[modality]['pval_df'] = pval_df
        
        # Store correlation and p-value data
        all_correlation_data[modality] = corr_df
        all_pvalue_data[modality] = pval_df
        
        # Collect all metadata variables
        all_metadata_vars.update(corr_df.index)
    
    return all_correlation_data, all_pvalue_data, all_metadata_vars, results


def _determine_shared_variable_order(all_metadata_vars, all_correlation_data, prioritized_vars, 
                                    max_metadata_vars, max_label_length, modalities):
    """Determine the shared variable order across modalities."""
    # Calculate max correlation for each variable across all modalities
    max_corr_data = _calculate_max_correlations(all_metadata_vars, all_correlation_data, modalities)
    
    # Create DataFrame and sort
    max_corr_df = pd.DataFrame.from_dict(max_corr_data, orient='index', columns=['MaxAbsCorr'])
    
    # Prioritize variables and then sort by max correlation
    shared_var_order = _prioritize_and_sort_variables(
        max_corr_df, prioritized_vars, all_metadata_vars, max_metadata_vars)
    
    # Create truncated names mapping if needed
    truncated_names = _create_truncated_names(shared_var_order, max_label_length)
    
    return shared_var_order, truncated_names


def _calculate_max_correlations(all_metadata_vars, all_correlation_data, modalities):
    """Calculate the maximum absolute correlation for each variable across all modalities."""
    max_corr_data = {}
    for var in all_metadata_vars:
        max_abs_corr = 0
        for modality in modalities:
            if modality in all_correlation_data and var in all_correlation_data[modality].index:
                var_max_corr = all_correlation_data[modality].loc[var].abs().max()
                max_abs_corr = max(max_abs_corr, var_max_corr)
        max_corr_data[var] = max_abs_corr
    return max_corr_data


def _prioritize_and_sort_variables(max_corr_df, prioritized_vars, all_metadata_vars, max_metadata_vars):
    """Prioritize specified variables and sort others by correlation strength."""
    prioritized_vars_set = set(prioritized_vars)
    existing_prioritized = [var for var in prioritized_vars if var in all_metadata_vars]
    other_vars = [var for var in all_metadata_vars if var not in prioritized_vars_set]
    
    # Sort other variables by max correlation
    if other_vars:
        other_vars_sorted = max_corr_df.loc[other_vars].sort_values('MaxAbsCorr', ascending=False).index
        # Calculate how many other vars we can show after accounting for prioritized vars
        remaining_slots = max_metadata_vars - len(existing_prioritized)
        # Take top N non-prioritized variables
        top_other_vars = other_vars_sorted[:min(remaining_slots, len(other_vars_sorted))]
        # Combine prioritized and other variables
        shared_var_order = existing_prioritized + list(top_other_vars)
    else:
        shared_var_order = existing_prioritized
        
    return shared_var_order


def _create_truncated_names(variable_list, max_label_length):
    """Create mapping for truncated variable names if needed."""
    truncated_names = {}
    if max_label_length > 0:
        for var in variable_list:
            if len(var) > max_label_length:
                truncated_names[var] = var[:max_label_length-3] + "..."
    return truncated_names


def _create_plots_for_each_modality(modalities, all_correlation_data, all_pvalue_data, 
                                  shared_var_order, truncated_names, plot_grid, gs,
                                  corr_method, cmap, size_scale, pval_threshold, results):
    """Create correlation plots for each modality."""
    for idx, modality in enumerate(modalities):
        if modality not in all_correlation_data:
            continue
        
        # Filter data to variables in the shared order
        corr_df_filtered, pval_df_filtered, display_vars = _prepare_modality_data(
            modality, all_correlation_data, all_pvalue_data, 
            shared_var_order, truncated_names)
        
        # Visualize correlations if we have room
        if idx < plot_grid[0] * plot_grid[1]:
            # Create subplot
            row, col = idx // plot_grid[1], idx % plot_grid[1]
            ax = plt.subplot(gs[row, col])
            
            # Determine if this is a leftmost plot in its row
            is_leftmost = (col == 0)
            
            # Create the plot
            subfig, plot_df = _plot_pc_metadata_correlation_dotplot(
                corr_df_filtered, pval_df_filtered, cmap, size_scale, None,
                pval_threshold, corr_method, modality, ax, None,
                y_order=display_vars,  # Pass the ordered y-axis labels
                show_y_axis=is_leftmost  # Only show y-axis for leftmost plots
            )
            results[modality]['plot_df'] = plot_df
        else:
            print(f"Warning: Not enough subplots for modality '{modality}'")


def _prepare_modality_data(modality, all_correlation_data, all_pvalue_data, 
                          shared_var_order, truncated_names):
    """Prepare data for a specific modality."""
    # Filter correlation and p-value matrices to shared order
    # Only include variables that exist in this modality
    modality_vars = [var for var in shared_var_order if var in all_correlation_data[modality].index]
    corr_df_filtered = all_correlation_data[modality].loc[modality_vars]
    pval_df_filtered = all_pvalue_data[modality].loc[modality_vars]
    
    # Apply truncated names if needed
    if truncated_names:
        # Create mapping for just this modality's variables
        mod_truncated_names = {var: truncated_names[var] for var in truncated_names if var in modality_vars}
        if mod_truncated_names:
            corr_df_filtered = corr_df_filtered.rename(index=mod_truncated_names)
            pval_df_filtered = pval_df_filtered.rename(index=mod_truncated_names)
    
    # Get the display names (potentially truncated)
    display_vars = [truncated_names.get(var, var) for var in modality_vars]
    
    return corr_df_filtered, pval_df_filtered, display_vars


def _add_colorbar_and_finalize(fig, gs, cmap, corr_method, save):
    """Add colorbar and finalize the figure."""
    # Create colorbar in the last column, spanning all rows
    cbar_ax = plt.subplot(gs[:, -1])  # Use the narrow last column
    norm = plt.Normalize(-1, 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"{corr_method.capitalize()} Correlation", size=12)
    
    # Add main title with significantly more space
    plt.suptitle("Metadata Correlations with Principal Components", 
                 fontsize=16, y=0.95)  # Positioned higher (y=0.95)
    
    # Save the figure if requested
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches="tight")


def _compute_pca_if_needed(adata: AnnData, **pca_kwargs) -> AnnData:
    """
    Compute PCA if not already present in the AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    **pca_kwargs
        Additional arguments to pass to sc.pp.pca()
    
    Returns
    -------
    AnnData
        The input AnnData with PCA computed
    """
    # Check if PCA is already computed
    if 'X_pca' not in adata.obsm.keys():
        sc.pp.pca(adata, **pca_kwargs)
    return adata


def _preprocess_metadata_for_correlation(
    adata: AnnData, 
    metadata_vars: Optional[List[str]] = None, 
    obsm_key: str = "X_pca", 
    n_pcs: int = 10,
    min_values_threshold: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess AnnData object to extract PCs and prepare metadata for correlation analysis.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing PCA results in adata.obsm[obsm_key]
    metadata_vars : list, optional
        List of metadata variables to correlate. If None, uses all categorical and numeric columns.
    obsm_key : str, optional
        Key in adata.obsm containing PCA coordinates (default: 'X_pca')
    n_pcs : int, optional
        Number of principal components to include (default: 10)
    min_values_threshold : float, optional
        Minimum fraction of non-missing values required (default: 0.5)
        
    Returns
    -------
    tuple
        (pc_df, metadata_df) containing preprocessed PCs and metadata
    """
    # Check if PCA results exist
    if obsm_key not in adata.obsm:
        raise ValueError(f"PCA results not found in adata.obsm['{obsm_key}']. Run sc.pp.pca() first.")
    
    # Extract PCA coordinates
    n_pcs = min(n_pcs, adata.obsm[obsm_key].shape[1])
    pc_df = pd.DataFrame(
        adata.obsm[obsm_key][:, :n_pcs],
        index=adata.obs_names,
        columns=[f"PC{i+1}" for i in range(n_pcs)]
    )
    
    # Determine metadata variables to use
    if metadata_vars is None:
        # Automatically select categorical and numeric columns
        metadata_vars = []
        for col in adata.obs.columns:
            # Skip columns with too many unique values (likely cell IDs or similar)
            if adata.obs[col].nunique() < min(100, adata.n_obs // 10):
                metadata_vars.append(col)
    
    # Select only the requested metadata
    metadata = adata.obs[metadata_vars].copy()
    
    # Convert categorical variables to numeric
    numeric_metadata = pd.DataFrame(index=metadata.index)
    
    for col in metadata.columns:
        if pd.api.types.is_categorical_dtype(metadata[col]):
            # One-hot encode categorical variables
            dummies = pd.get_dummies(metadata[col], prefix=col)
            # Keep original column name for binary variables
            if dummies.shape[1] == 2:
                dummies = dummies.iloc[:, [1]]
                dummies.columns = [col]
            numeric_metadata = pd.concat([numeric_metadata, dummies], axis=1)
        elif pd.api.types.is_numeric_dtype(metadata[col]):
            # Add directly if already numeric
            numeric_metadata[col] = metadata[col]
    
    # Filter out variables with too many missing values
    valid_cols = []
    for col in numeric_metadata.columns:
        # Calculate fraction of non-missing values
        non_missing_fraction = numeric_metadata[col].notna().mean()
        
        # Keep variables with sufficient non-missing values
        if non_missing_fraction >= min_values_threshold:
            valid_cols.append(col)
        else:
            print(f"Filtering out '{col}': only {non_missing_fraction:.1%} non-missing values (threshold: {min_values_threshold:.0%})")
    
    # Filter metadata to keep only columns with sufficient data
    filtered_metadata = numeric_metadata[valid_cols]
    
    return pc_df, filtered_metadata


def _calculate_pc_metadata_correlations(
    pc_df: pd.DataFrame, 
    metadata_df: pd.DataFrame, 
    corr_method: str = "spearman"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate correlations between metadata variables and principal components.
    
    Parameters
    ----------
    pc_df : pandas.DataFrame
        DataFrame containing principal component values
    metadata_df : pandas.DataFrame
        DataFrame containing metadata variables
    corr_method : str, optional
        Correlation method: 'spearman' or 'pearson' (default: 'spearman')
        
    Returns
    -------
    tuple
        (correlation_df, pvalue_df) DataFrames containing correlation values and p-values
    """
    # Set correlation function
    if corr_method == "spearman":
        corr_func = spearmanr
    elif corr_method == "pearson":
        corr_func = pearsonr
    else:
        raise ValueError("corr_method must be 'spearman' or 'pearson'")
    
    # Initialize matrices for correlation values and p-values
    corr_values = np.zeros((len(metadata_df.columns), len(pc_df.columns)))
    p_values = np.zeros((len(metadata_df.columns), len(pc_df.columns)))
    
    # Calculate correlations and p-values
    for i, meta_col in enumerate(metadata_df.columns):
        meta_values = metadata_df[meta_col].values
        
        for j, pc_col in enumerate(pc_df.columns):
            pc_values = pc_df[pc_col].values
            
            # Calculate correlation only for non-missing values
            mask = ~np.isnan(meta_values) & ~np.isnan(pc_values)
            
            # Calculate the number of valid sample pairs
            valid_sample_count = np.sum(mask)
            
            # Require at least 10 valid samples
            min_samples = 10  # Increased from 5 to 10
            
            if valid_sample_count >= min_samples:
                corr, pval = corr_func(meta_values[mask], pc_values[mask])
                if hasattr(corr, "__len__"):  # spearmanr can return matrix for some versions
                    corr = corr[0]
                    pval = pval[0]
            else:
                corr, pval = np.nan, np.nan
                print(f"Skipping correlation for '{meta_col}' and {pc_col}: only {valid_sample_count} valid samples (threshold: {min_samples})")
            
            corr_values[i, j] = corr
            p_values[i, j] = pval
    
    # Create DataFrames
    corr_df = pd.DataFrame(
        corr_values,
        index=metadata_df.columns,
        columns=pc_df.columns
    )
    
    pval_df = pd.DataFrame(
        p_values,
        index=metadata_df.columns,
        columns=pc_df.columns
    )
    
    return corr_df, pval_df


def _plot_pc_metadata_correlation_dotplot(
    corr_df: pd.DataFrame, 
    pval_df: pd.DataFrame, 
    cmap: str = "RdBu_r", 
    size_scale: float = 50, 
    figsize: Optional[Tuple[float, float]] = (10, 8), 
    pval_threshold: float = 0.05, 
    corr_method: str = "spearman",
    modality: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save: Optional[str] = None,
    y_order: Optional[List[str]] = None,
    show_y_axis: bool = True
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Create a dot plot showing correlations between metadata variables and principal components.
    
    Parameters
    ----------
    corr_df : pandas.DataFrame
        DataFrame containing correlation values
    pval_df : pandas.DataFrame
        DataFrame containing p-values
    cmap : str, optional
        Colormap for correlation values (default: 'RdBu_r')
    size_scale : float, optional
        Scaling factor for dot sizes (default: 50)
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (10, 8))
    pval_threshold : float, optional
        Threshold for statistical significance (default: 0.05)
    corr_method : str, optional
        Correlation method used (for plot title)
    modality : str, optional
        Name of the modality (for plot title)
    ax : matplotlib.Axes, optional
        Axes to plot on. If None, creates new figure
    save : str, optional
        Path to save the figure. If None, the figure is not saved.
    y_order : list, optional
        Order of y-axis categories (metadata variables)
    show_y_axis : bool, optional
        Whether to show y-axis labels and tick marks (default: True)
        
    Returns
    -------
    tuple
        (fig, plot_df) containing the figure and plotting data
    """
    # Prepare data for plotting
    plot_df = _prepare_plotting_data(corr_df, pval_df, pval_threshold, size_scale)
    
    # Set up figure and axes
    fig, ax = _setup_plot_axes(ax, figsize)
    
    # Create the scatterplot
    g = _create_scatterplot(plot_df, cmap, size_scale, ax)
    
    # Configure y-axis ordering and visibility
    _configure_y_axis(ax, y_order, show_y_axis)
    
    # Set plot titles and labels
    _set_plot_titles_and_labels(ax, corr_method, modality)
    
    # Handle colorbar if needed
    if ax is None or fig != plt.gcf():  # Only add colorbar for standalone plots
        _add_standalone_colorbar(ax, cmap, corr_method)
    
    # Remove the size legend
    _remove_size_legend(g)
    
    # Apply tight layout for standalone figures
    if ax is None or fig != plt.gcf():
        plt.tight_layout()
    
    # Save the figure if requested
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    
    return fig, plot_df


def _prepare_plotting_data(corr_df, pval_df, pval_threshold, size_scale):
    """Prepare data for plotting by converting to long format."""
    data = []
    
    for meta in corr_df.index:
        for pc in corr_df.columns:
            corr = corr_df.loc[meta, pc]
            pval = pval_df.loc[meta, pc]
            sig = bool(pval <= pval_threshold)  # Explicitly convert to bool
            
            data.append({
                "Metadata": meta,
                "PC": pc,
                "Correlation": corr,
                "P-value": pval,
                "Significant": sig,
                "Size": abs(corr) * size_scale
            })
    
    # Create DataFrame explicitly from list of dicts to avoid warnings
    return pd.DataFrame(data)


def _setup_plot_axes(ax, figsize):
    """Set up the plotting axes."""
    create_new_fig = ax is None
    if create_new_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    return fig, ax


def _create_scatterplot(plot_df, cmap, size_scale, ax):
    """Create the scatterplot visualization with auto-scaling correlation range."""
    # Automatically determine appropriate correlation scale
    max_abs_corr = plot_df["Correlation"].abs().max()
    
    # Set scale based on maximum correlation value
    if max_abs_corr < 0.1:
        corr_range = (-0.1, 0.1)
    elif max_abs_corr < 0.2:
        corr_range = (-0.2, 0.2)
    elif max_abs_corr < 0.5:
        corr_range = (-0.5, 0.5)
    else:
        corr_range = (-1, 1)
    
    # Create and return the scatterplot with appropriate scale
    return sns.scatterplot(
        data=plot_df,
        x="PC",
        y="Metadata",
        size="Size",
        hue="Correlation",
        palette=cmap,
        sizes=(0, size_scale),
        hue_norm=corr_range,  # Use auto-determined range
        edgecolor='black',
        linewidth=0.3,
        ax=ax
    )


def _configure_y_axis(ax, y_order, show_y_axis):
    """Configure y-axis ordering and visibility."""
    # Set y-axis order if provided
    if y_order is not None:
        # Get current y-tick positions
        yticks = list(range(len(y_order)))
        # Set the y-ticks and labels
        ax.set_yticks(yticks)
        ax.set_yticklabels(y_order)
        # Ensure the order is maintained
        ax.set_ylim(len(y_order) - 0.5, -0.5)  # Reverse the y-axis for better visualization
    
    # Hide y-axis if not a leftmost plot
    if not show_y_axis:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', left=False)
    else:
        ax.set_ylabel("Sample Metadata", labelpad=10)


def _set_plot_titles_and_labels(ax, corr_method, modality):
    """Set plot titles and axis labels."""
    title = f"{corr_method.capitalize()} Correlation"
    if modality:
        title += f" - {modality.capitalize()} Modality"
    
    ax.set_title(title, pad=20)  # Increased padding to move title up
    ax.set_xlabel("Principal Components", labelpad=10)


def _add_standalone_colorbar(ax, cmap, corr_method):
    """Add colorbar for standalone plots."""
    norm = plt.Normalize(-1, 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f"{corr_method.capitalize()} Correlation")


def _remove_size_legend(g):
    """Remove the size legend from the plot."""
    handles, labels = g.get_legend_handles_labels()
    if len(handles) > 0:
        try:
            g.legend_.remove()
        except (AttributeError, KeyError) as e:
            # Handle specific exceptions that might occur when legend doesn't exist
            # or has already been removed
            pass


# PC plots
def _format_value_for_legend(value):
    """Format a value for displaying in a legend."""
    if pd.isna(value):
        return "Missing/NaN"
    elif isinstance(value, bool):
        return str(value)
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        else:
            return f"{value:.2f}".rstrip('0').rstrip('.')
    else:
        return str(value)


def _create_category_mask(data, attribute, category):
    """Create a boolean mask for a specific category in the data."""
    if pd.isna(category):
        # Special handling for NaN values
        return data[attribute].isna()
    elif isinstance(category, (int, float, bool)):
        # For numeric types, use exact comparison
        return data[attribute] == category
    else:
        # For other types, use string comparison
        return data[attribute].astype(str) == str(category)


def _is_numeric_attribute(attr_series):
    """Determine if an attribute is numeric and has more than 2 unique values."""
    # Check if series is numeric
    if pd.api.types.is_numeric_dtype(attr_series):
        # Count unique non-NaN values
        unique_count = attr_series.dropna().nunique()
        return unique_count > 2
    return False