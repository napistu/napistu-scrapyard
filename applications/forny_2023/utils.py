from joblib import Parallel, delayed
from typing import Union, List, Optional, Tuple, Dict

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from anndata import AnnData
from mudata import MuData
from scipy.stats import spearmanr, pearsonr


# Set up the plotting style with a standard matplotlib style
plt.style.use("default")
sc.settings.set_figure_params(dpi=100, frameon=True)


def plot_mudata_pca_variance(mdata: MuData,
                            modalities: List[str] = None,
                            pca_kwargs: Dict = None,
                            plot_kwargs: Dict = None,
                            figsize: Tuple[float, float] = (12, 5),
                            suptitle: Optional[str] = 'PCA Variance Ratio by Modality') -> plt.Figure:
    """
    Create side-by-side PCA variance plots for multiple modalities in a MuData object.
    
    Parameters
    ----------
    mdata : MuData
        Multi-modal data object containing multiple AnnData objects
    modalities : List[str], optional
        List of modality names to plot, defaults to all modalities in mdata
    pca_kwargs : Dict, optional
        Additional arguments to pass to sc.pp.pca() for each modality
    plot_kwargs : Dict, optional
        Additional arguments to pass to _plot_pca_variance()
    figsize : Tuple[float, float], default=(12, 5)
        Figure size (width, height)
    suptitle : str, optional
        Super title for the figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    # Set defaults
    if modalities is None:
        modalities = list(mdata.mod.keys())
    
    if pca_kwargs is None:
        pca_kwargs = {}
    
    if plot_kwargs is None:
        plot_kwargs = {}
    
    # Set up color palettes for each modality
    color_palettes = {
        'transcriptomics': {'bar': '#1f77b4', 'line': '#ff7f0e', 'threshold': '#ff7f0e'},
        'proteomics': {'bar': '#2ca02c', 'line': '#d62728', 'threshold': '#d62728'},
        'spatial': {'bar': '#9467bd', 'line': '#8c564b', 'threshold': '#8c564b'},
        'chromatin': {'bar': '#e377c2', 'line': '#7f7f7f', 'threshold': '#7f7f7f'},
        # Add more modalities and color schemes as needed
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(modalities), figsize=figsize)
    if len(modalities) == 1:
        axes = [axes]
    
    # Process each modality
    for idx, modality in enumerate(modalities):
        if modality not in mdata.mod:
            raise ValueError(f"Modality '{modality}' not found in MuData object")
        
        # Compute PCA if needed
        _compute_pca_if_needed(mdata.mod[modality], **pca_kwargs)
        
        # Get colors for this modality
        colors = color_palettes.get(modality.lower(), 
                                   {'bar': '#1f77b4', 'line': '#ff7f0e', 'threshold': '#ff7f0e'})
        
        # Plot variance
        plot_args = plot_kwargs.copy() if plot_kwargs else {}
        plot_args['color_palette'] = colors
        plot_args['title'] = f'{modality.capitalize()}'
        _plot_pca_variance(mdata.mod[modality], ax=axes[idx], **plot_args)
    
    # Add super title
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    
    return fig


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
    max_label_length: int = 25
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
        mdata, modalities, obsm_key, n_pcs, metadata_vars, pca_kwargs, corr_method)
    
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


def process_and_plot_pca(adata, title):
    """
    Process data and create a PCA plot.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    title : str
        Title for the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated PCA plot figure
    """
    # Preprocessing
    sc.pp.scale(adata)
    
    # Run PCA
    sc.tl.pca(adata, n_comps=50)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get the case information and convert to categorical
    case_colors = ["#FF7F50" if x == 1 else "#4169E1" for x in adata.obs["case"]]
    
    # Scatter plot
    ax.scatter(
        adata.obsm["X_pca"][:, 0],
        adata.obsm["X_pca"][:, 1],
        c=case_colors,
        alpha=0.7,
        s=50,
    )
    
    # Add labels and title
    ax.set_xlabel(f'PC1 ({adata.uns["pca"]["variance_ratio"][0]:.1%} variance explained)')
    ax.set_ylabel(f'PC2 ({adata.uns["pca"]["variance_ratio"][1]:.1%} variance explained)')
    ax.set_title(f"PCA - {title}")
    
    # Add legend
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FF7F50",
            label="Case (1)",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#4169E1",
            label="Control (0)",
            markersize=10,
        ),
    ]
    ax.legend(handles=legend_elements)
    
    # Add grid for better readability
    ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    return fig





def apply_regression_per_feature(
    adata, formula, covariates_df=None, layer=None, n_jobs=-1, batch_size=100, progress_bar=True
):
    """
    Apply linear regression to each feature in an AnnData object and return statistics.
    Supports parallel processing across multiple cores using joblib.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    formula : str
        Formula for regression in patsy format (e.g., '~ batch + n_counts').
        Don't include the dependent variable, as each feature will be used.
    covariates_df : pandas.DataFrame, optional
        DataFrame containing covariates. If None, uses adata.obs.
    layer : str, optional
        If provided, use this layer instead of X.
    n_jobs : int, optional
        Number of parallel jobs. -1 means using all processors.
    batch_size : int, optional
        Number of features to process in each batch.
    progress_bar : bool, optional
        Whether to display a progress bar.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with regression statistics for each feature.
    """
    # Set up covariates
    if covariates_df is None:
        covariates_df = adata.obs.copy()
    
    # Get data to regress
    if layer is None:
        if isinstance(adata.X, np.ndarray):
            X_data = adata.X
        else:
            X_data = adata.X.toarray()
    else:
        if isinstance(adata.layers[layer], np.ndarray):
            X_data = adata.layers[layer]
        else:
            X_data = adata.layers[layer].toarray()
    
    # Define a function to process a single feature
    def process_feature(i):
        feature_name = adata.var_names[i]
        y = X_data[:, i]
        
        # Skip features with zero variance
        if np.var(y) == 0:
            return []
        
        # Create temporary dataframe with feature and covariates
        temp_df = covariates_df.copy()
        temp_df["y"] = y
        
        feature_results = []
        
        try:
            # Full formula with the feature as dependent variable
            full_formula = "y " + formula
            model = sm.formula.ols(formula=full_formula, data=temp_df)
            result = model.fit()
            
            # Extract statistics for each coefficient
            for coef_name in result.params.index:
                if coef_name == "Intercept":
                    continue  # Skip intercept
                
                feature_results.append(
                    {
                        "feature": feature_name,
                        "coefficient": coef_name,
                        "estimate": result.params[coef_name],
                        "std_err": result.bse[coef_name],
                        "statistic": result.tvalues[coef_name],
                        "p_value": result.pvalues[coef_name],
                        "conf_int_lower": result.conf_int().loc[coef_name, 0],
                        "conf_int_upper": result.conf_int().loc[coef_name, 1],
                    }
                )
        except Exception as e:
            print(f"Error in feature {feature_name}: {str(e)}")
        
        return feature_results
    
    # Create feature indices
    feature_indices = list(range(X_data.shape[1]))
    
    # Verbose setting for progress bar
    verbose = 10 if progress_bar else 0
    
    # Run parallel processing with joblib
    print(f"Starting regression analysis with {n_jobs} cores...")
    results_nested = Parallel(n_jobs=n_jobs, batch_size=batch_size, verbose=verbose)(
        delayed(process_feature)(i) for i in feature_indices
    )
    
    # Flatten results
    results = []
    for sublist in results_nested:
        results.extend(sublist)
    
    # Handle empty results
    if not results:
        print("No regression results were generated. Check your data and formula.")
        return pd.DataFrame()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add FDR correction (Benjamini-Hochberg)
    if not results_df.empty and "p_value" in results_df.columns:
        # Group by coefficient and calculate FDR-corrected p-values
        coefs = results_df["coefficient"].unique()
        for coef in coefs:
            mask = results_df["coefficient"] == coef
            results_df.loc[mask, "fdr_bh"] = sm.stats.multipletests(
                results_df.loc[mask, "p_value"], method="fdr_bh"
            )[1]
    
    print(f"Completed regression analysis for {len(results_df)} feature-coefficient pairs.")
    return results_df


def add_regression_results_to_anndata(
    adata, results_df, key_added="regression_results", fdr_cutoff=0.05, inplace=True
):
    """
    Add regression results dataframe to an AnnData object in a structured way.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    results_df : pandas.DataFrame
        DataFrame with regression results as produced by apply_regression_per_feature().
    key_added : str, optional
        Key under which to add the results in adata.uns.
    fdr_cutoff : float, optional
        Significance threshold for FDR-corrected p-values. Default is 0.05.
    inplace : bool, optional
        If True, modify adata inplace. Otherwise return a copy. Default is True.
        
    Returns
    -------
    Depends on `inplace` parameter. If `inplace=True`, returns None,
    otherwise returns a modified copy of the AnnData object.
    """
    if not inplace:
        adata = adata.copy()
    
    # Make a copy to avoid modifying the original results
    results = results_df.copy()
    
    # Create a hierarchical dictionary to store in adata.uns
    regression_dict = {
        "params": {
            "features": results["feature"].unique().tolist(),
            "coefficients": results["coefficient"].unique().tolist(),
            "n_features": len(results["feature"].unique()),
            "n_coefficients": len(results["coefficient"].unique()),
            "fdr_cutoff": fdr_cutoff,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results.to_dict("records"),
    }
    
    # Store in adata.uns
    adata.uns[key_added] = regression_dict
    
    # Also annotate significant associations in var
    # Create a significance mask for each coefficient (using FDR if available)
    for coef in results["coefficient"].unique():
        coef_safe = coef.replace(" ", "_").replace("-", "_")
        
        # Determine the p-value column to use (FDR-corrected if available)
        p_col = "fdr_bh" if "fdr_bh" in results.columns else "p_value"
        
        # Get significant features for this coefficient
        sig_mask = (results["coefficient"] == coef) & (results[p_col] < fdr_cutoff)
        sig_features = results.loc[sig_mask, "feature"].unique()
        
        # Create a binary indicator in adata.var
        adata.var[f"sig_{coef_safe}"] = adata.var_names.isin(sig_features)
        
        # Add effect size for significant features
        effect_dict = {}
        for _, row in results[results["coefficient"] == coef].iterrows():
            effect_dict[row["feature"]] = row["estimate"]
        
        # Create an effect size column (NaN for non-significant features)
        adata.var[f"effect_{coef_safe}"] = pd.Series(
            [
                effect_dict.get(f, np.nan) if f in sig_features else np.nan
                for f in adata.var_names
            ],
            index=adata.var_names,
        )
        
        # Also add the p-values as a separate column
        pval_dict = {}
        for _, row in results[results["coefficient"] == coef].iterrows():
            pval_dict[row["feature"]] = row[p_col]
            
        adata.var[f"pval_{coef_safe}"] = pd.Series(
            [pval_dict.get(f, np.nan) for f in adata.var_names], index=adata.var_names
        )
    
    # Add a summarized view for quick access to top features per coefficient
    top_features = {}
    for coef in results["coefficient"].unique():
        # Sort by absolute effect size
        coef_results = results[results["coefficient"] == coef].copy()
        coef_results["abs_effect"] = np.abs(coef_results["estimate"])
        coef_results = coef_results.sort_values("abs_effect", ascending=False)
        
        # Get top 50 features by effect size (with p-value < fdr_cutoff)
        p_col = "fdr_bh" if "fdr_bh" in coef_results.columns else "p_value"
        top = coef_results[coef_results[p_col] < fdr_cutoff].head(50)
        
        if len(top) > 0:
            top_features[coef] = top[["feature", "estimate", p_col]].to_dict("records")
    
    # Store top features in uns
    adata.uns[f"{key_added}_top_features"] = top_features
    
    return None if inplace else adata


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


def _plot_pca_variance(adata: AnnData, 
                      n_pcs: Optional[int] = None,
                      log: bool = False,
                      show_cum_var: bool = True,
                      cum_var_threshold: float = 0.8,
                      ax: Optional[plt.Axes] = None,
                      title: Optional[str] = None,
                      color_palette: Optional[Dict] = None) -> Tuple[plt.Axes, Optional[plt.Axes]]:
    """
    Plot the PCA variance ratio and optionally the cumulative variance.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with PCA computed
    n_pcs : int, optional
        Number of PCs to plot, defaults to all available
    log : bool, default=False
        Whether to use log scale for y-axis
    show_cum_var : bool, default=True
        Whether to show cumulative variance as a line plot
    cum_var_threshold : float, default=0.8
        Threshold to show on cumulative variance plot
    ax : matplotlib.Axes, optional
        Axes to plot on, new axes created if None
    title : str, optional
        Title for the plot
    color_palette : dict, optional
        Dictionary with keys 'bar', 'line', 'threshold' for colors
        
    Returns
    -------
    tuple
        Main axes and secondary axes (if cumulative variance is shown)
    """
    # Set default colors
    if color_palette is None:
        color_palette = {
            'bar': '#1f77b4',
            'line': '#ff7f0e',
            'threshold': '#ff7f0e'
        }
    
    # Get variance ratio data
    var_ratio = adata.uns['pca']['variance_ratio']
    
    if n_pcs is None:
        n_pcs = len(var_ratio)
    else:
        n_pcs = min(n_pcs, len(var_ratio))
    
    var_ratio = var_ratio[:n_pcs]
    
    # Create new axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot variance explained (bar plot)
    ax.bar(np.arange(1, n_pcs+1), var_ratio, color=color_palette['bar'], 
           alpha=0.7, label='Variance explained')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Ratio')
    
    if title:
        ax.set_title(title)
    
    if log:
        ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks to be integers
    ax.set_xticks(np.arange(1, n_pcs+1, max(1, n_pcs//10)))
    
    # Show cumulative variance if requested
    ax2 = None
    if show_cum_var:
        ax2 = ax.twinx()
        cum_var = np.cumsum(var_ratio)
        ax2.plot(np.arange(1, n_pcs+1), cum_var, 
                color=color_palette['line'], marker='o', markersize=4,
                label='Cumulative variance')
        ax2.set_ylabel('Cumulative Variance Ratio', color=color_palette['line'])
        ax2.tick_params(axis='y', labelcolor=color_palette['line'])
        ax2.set_ylim([0, min(1.05, max(cum_var) * 1.1)])
        
        # Add horizontal line at threshold
        if cum_var_threshold is not None:
            ax2.axhline(y=cum_var_threshold, color=color_palette['threshold'], 
                       linestyle='--', alpha=0.5,
                       label=f'{cum_var_threshold:.0%} threshold')
            
            # Find where cumulative variance exceeds threshold
            threshold_idx = np.argmax(cum_var >= cum_var_threshold)
            if threshold_idx < len(cum_var):
                threshold_pc = threshold_idx + 1
                ax2.annotate(f'PC{threshold_pc}', 
                            xy=(threshold_pc, cum_var_threshold),
                            xytext=(threshold_pc + 1, cum_var_threshold - 0.1),  # Positioning from lower right
                            arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-0.2"),  # Curved arrow
                            horizontalalignment='left',
                            verticalalignment='top')
    
    return ax, ax2

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


def _process_all_modalities(mdata, modalities, obsm_key, n_pcs, metadata_vars, pca_kwargs, corr_method):
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
            adata, metadata_vars, obsm_key, n_pcs
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
    n_pcs: int = 10
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
    
    return pc_df, numeric_metadata


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
            if sum(mask) > 5:  # Require at least 5 valid samples
                corr, pval = corr_func(meta_values[mask], pc_values[mask])
                if hasattr(corr, "__len__"):  # spearmanr can return matrix for some versions
                    corr = corr[0]
                    pval = pval[0]
            else:
                corr, pval = np.nan, np.nan
            
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
    """Create the scatterplot visualization."""
    return sns.scatterplot(
        data=plot_df,
        x="PC",
        y="Metadata",
        size="Size",
        hue="Correlation",
        palette=cmap,
        sizes=(0, size_scale),
        hue_norm=(-1, 1),
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