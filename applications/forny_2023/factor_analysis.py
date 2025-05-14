"""
MOFA Factor Analysis Utilities

This module provides functions for running and analyzing Multi-Omics Factor Analysis (MOFA)
with different numbers of factors.
"""

from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional, Iterable, Any, Union

import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
from mudata import MuData
import muon
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

FACTOR_REGRESSION_STR = "mofa_regression_{}"

def create_minimal_mudata(
    original_mdata: md.MuData,
    include_layers: Optional[List[str]] = None,
    include_obsm: bool = True,
    include_varm: bool = False,
) -> md.MuData:
    """
    Create a minimal copy of a MuData object with only essential components
    to avoid serialization issues.
    
    Parameters
    ----------
    original_mdata : md.MuData
        Original MuData object to copy
    include_layers : Optional[List[str]]
        List of layer names to include in the copy, by default None
    include_obsm : bool
        Whether to include obsm matrices, by default True
    include_varm : bool
        Whether to include varm matrices, by default False
        
    Returns
    -------
    md.MuData
        Minimal MuData object
        
    Examples
    --------
    >>> # Create a minimal copy with just the log-normalized layer
    >>> minimal_mdata = create_minimal_mudata(
    ...     mdata, 
    ...     include_layers=["log_normalized"], 
    ...     include_obsm=True
    ... )
    """
    # Create dictionary to hold modalities
    modalities = {}
    
    # Copy over each modality with minimal data
    for mod_name, mod in original_mdata.mod.items():
        # Create basic AnnData with just X, obs, and var
        new_mod = ad.AnnData(
            X=mod.X.copy(),
            obs=mod.obs.copy(),
            var=mod.var.copy()
        )
        
        # Add selected layers if present
        if include_layers and hasattr(mod, "layers"):
            for layer_name in include_layers:
                if layer_name in mod.layers:
                    new_mod.layers[layer_name] = mod.layers[layer_name].copy()
        
        # Add obsm matrices if requested
        if include_obsm and hasattr(mod, "obsm"):
            for obsm_key in mod.obsm.keys():
                new_mod.obsm[obsm_key] = mod.obsm[obsm_key].copy()
        
        # Add varm matrices if requested
        if include_varm and hasattr(mod, "varm"):
            for varm_key in mod.varm.keys():
                new_mod.varm[varm_key] = mod.varm[varm_key].copy()
        
        # Add to modalities dictionary
        modalities[mod_name] = new_mod
    
    # Create MuData object with dictionary of modalities
    minimal_mdata = md.MuData(modalities)
    
    # Copy global obsm if present and requested
    if include_obsm and hasattr(original_mdata, "obsm"):
        for obsm_key in original_mdata.obsm.keys():
            minimal_mdata.obsm[obsm_key] = original_mdata.obsm[obsm_key].copy()
    
    return minimal_mdata


def run_mofa_factor_scan(
    mdata: md.MuData,
    factor_range: Iterable[int] = range(5, 31, 5),
    use_layer: Optional[str] = "log2_centered",
    use_var: Optional[str] = None,
    seed: int = 42,
    models_dir: str = "mofa_models",
    overwrite: bool = False,
) -> Dict[int, Dict[str, str]]:
    """
    Run MOFA for multiple factor values and save results to files.
    
    Parameters
    ----------
    mdata : md.MuData
        Multi-modal data object
    factor_range : Iterable[int]
        Range of factors to test, by default range(5, 31, 5)
    use_layer : Optional[str]
        Layer to use for analysis, by default "log2_centered"
    use_var : Optional[str]
        Variable selection to use, by default None
    seed : int
        Random seed for reproducibility, by default 42
    models_dir : str
        Directory to store result files, by default "mofa_models"
    overwrite : bool
        Whether to overwrite existing results, by default False
        
    Returns
    -------
    Dict[int, Dict[str, str]]
        Dictionary with file paths for each factor value
        
    Examples
    --------
    >>> # Run MOFA with different numbers of factors
    >>> factor_results = run_mofa_factor_scan(
    ...     mdata, 
    ...     factor_range=range(5, 51, 5),
    ...     use_layer="log2_centered",
    ...     models_dir="mofa_results"
    ... )
    """
    # Create directory to store MOFA results
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {}
    
    # Run MOFA for each factor value
    for n_factors in factor_range:
        # Define paths for output files
        variance_file = os.path.join(models_dir, f"mofa_{n_factors}_factors_variance.npz")
        factors_file = os.path.join(models_dir, f"mofa_{n_factors}_factors_scores.npy")
        
        # Skip if files exist and we're not overwriting
        if not overwrite and os.path.exists(variance_file) and os.path.exists(factors_file):
            print(f"Results for {n_factors} factors already exist.")
            results[n_factors] = {
                "variance_file": variance_file,
                "factors_file": factors_file
            }
            continue
        
        print(f"[{_get_timestamp()}] Running MOFA with {n_factors} factors...")
        
        # Create a minimal copy to avoid serialization issues
        clean_mdata = create_minimal_mudata(
            mdata, 
            include_layers=[use_layer] if use_layer else None
        )
        
        try:
            # Run MOFA without saving to file
            muon.tl.mofa(
                clean_mdata,
                n_factors=n_factors,
                use_obs=None,
                use_var=use_var,
                use_layer=use_layer,
                outfile=None,  # Don't save to file
                seed=seed,
                convergence_mode="fast",
                verbose=False,
                save_metadata=True
            )
            
            # Extract variance information
            if "mofa" in clean_mdata.uns and "variance" in clean_mdata.uns["mofa"]:
                variance_data = clean_mdata.uns["mofa"]["variance"]
                
                # Save variance data
                np.savez(variance_file, **variance_data)
                
                # Save factor scores if available
                if "X_mofa" in clean_mdata.obsm:
                    factor_scores = clean_mdata.obsm["X_mofa"]
                    np.save(factors_file, factor_scores)
                
                # Store file paths in results
                results[n_factors] = {
                    "variance_file": variance_file,
                    "factors_file": factors_file
                }
                
                print(f"[{_get_timestamp()}] Saved results for {n_factors} factors.")
            else:
                print(f"No MOFA results found for {n_factors} factors.")
                
        except Exception as e:
            print(f"[{_get_timestamp()}] Error with {n_factors} factors: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return results


def calculate_variance_metrics(
    results_dict: Dict[int, Dict[str, str]]
) -> Dict[int, Dict[str, Any]]:
    """
    Calculate variance metrics from saved MOFA results.
    
    Parameters
    ----------
    results_dict : Dict[int, Dict[str, str]]
        Dictionary with file paths for each factor value
        
    Returns
    -------
    Dict[int, Dict[str, Any]]
        Dictionary with variance metrics for each factor value
        
    Examples
    --------
    >>> # Calculate variance metrics from saved results
    >>> metrics = calculate_variance_metrics(factor_results)
    >>> # Print total variance for each number of factors
    >>> for n_factors, data in metrics.items():
    ...     print(f"{n_factors} factors: {data['total_variance']:.4f}")
    """
    metrics = {}
    
    for n_factors, paths in results_dict.items():
        try:
            variance_file = paths["variance_file"]
            
            # Load variance data
            variance_data = dict(np.load(variance_file))
            
            # Calculate metrics
            modality_variance = {k: np.sum(v) for k, v in variance_data.items()}
            total_variance = sum(modality_variance.values())
            
            # Accumulate per-factor variance
            per_factor_variance = _calculate_per_factor_variance(variance_data)
            
            # Store metrics
            metrics[n_factors] = {
                "total_variance": total_variance,
                "modality_variance": modality_variance,
                "per_factor_variance": per_factor_variance.tolist() if per_factor_variance is not None else None,
                "factor_view_variance": {k: v.tolist() for k, v in variance_data.items()}
            }
            
            print(f"Calculated metrics for {n_factors} factors. Total variance: {total_variance:.4f}")
            
        except Exception as e:
            print(f"Error calculating metrics for {n_factors} factors: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return metrics


def determine_optimal_factors(
    metrics: Dict[int, Dict[str, Any]], 
    criterion: str = "elbow", 
    threshold: float = 0.01
) -> Optional[int]:
    """
    Determine the optimal number of factors based on variance metrics.
    
    Parameters
    ----------
    metrics : Dict[int, Dict[str, Any]]
        Dictionary with variance metrics for each factor value
    criterion : str
        Method to use for determining optimal factors
        ('elbow', 'threshold', or 'balanced'), by default "elbow"
    threshold : float
        Threshold for marginal improvement (used if criterion='threshold'),
        by default 0.01
        
    Returns
    -------
    Optional[int]
        Optimal number of factors or None if it cannot be determined
        
    Examples
    --------
    >>> # Determine optimal factors using different criteria
    >>> optimal_elbow = determine_optimal_factors(metrics, criterion='elbow')
    >>> optimal_threshold = determine_optimal_factors(metrics, criterion='threshold', threshold=0.01)
    >>> optimal_balanced = determine_optimal_factors(metrics, criterion='balanced')
    >>> print(f"Optimal factors: elbow={optimal_elbow}, threshold={optimal_threshold}, balanced={optimal_balanced}")
    """
    # Extract factors and variance values
    factors = sorted([k for k in metrics.keys() if metrics[k]["total_variance"] is not None])
    
    if not factors:
        print("No valid variance metrics available")
        return None
        
    variance = [metrics[k]["total_variance"] for k in factors]

    if len(factors) <= 1:
        return factors[0] if factors else None

    # Calculate marginal improvements
    diffs = np.diff(variance)

    if criterion == "elbow":
        return _determine_optimal_elbow(factors, variance, diffs)
    elif criterion == "threshold":
        return _determine_optimal_threshold(factors, diffs, threshold)
    elif criterion == "balanced":
        return _determine_optimal_balanced(factors, variance)
    else:
        raise ValueError(f"Unknown criterion: {criterion}. Use 'elbow', 'threshold', or 'balanced'.")


def visualize_variance_explained(
    metrics: Dict[int, Dict[str, Any]],
    optimal_factors: Optional[Dict[str, int]] = None,
    user_factors: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a plot of total variance explained by number of factors.
    
    Parameters
    ----------
    metrics : Dict[int, Dict[str, Any]]
        Dictionary with variance metrics for each factor value
    optimal_factors : Optional[Dict[str, int]]
        Dictionary mapping method names to optimal factor values, by default None
    user_factors : Optional[int]
        User-specified number of factors to highlight, by default None
    figsize : Tuple[int, int]
        Figure size, by default (12, 8)
    save_path : Optional[str]
        Path to save the figure, by default None
        
    Returns
    -------
    plt.Figure
        Created figure
        
    Examples
    --------
    >>> # Calculate optimal factors using different criteria
    >>> optimal_factors = {
    ...     "elbow": determine_optimal_factors(metrics, criterion='elbow'),
    ...     "threshold": determine_optimal_factors(metrics, criterion='threshold'),
    ...     "balanced": determine_optimal_factors(metrics, criterion='balanced')
    ... }
    >>> # Visualize variance explained with optimal factors highlighted
    >>> fig = visualize_variance_explained(metrics, optimal_factors, user_factors=15)
    >>> plt.show()
    """
    # Extract factors and variance values
    factors = sorted([k for k in metrics.keys() if metrics[k]["total_variance"] is not None])
    
    if not factors:
        print("No valid variance metrics available")
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "No valid variance metrics available", 
                 horizontalalignment="center", verticalalignment="center")
        return fig
        
    variance = [metrics[k]["total_variance"] for k in factors]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Plot total variance explained
    plt.plot(factors, variance, "o-", linewidth=2, color="blue", markersize=8)
    
    # Highlight optimal factors if provided
    colors = {"elbow": "red", "threshold": "green", "balanced": "purple"}
    
    if optimal_factors:
        for method, n_factors in optimal_factors.items():
            if n_factors in factors:
                color = colors.get(method, "gray")
                idx = factors.index(n_factors)
                plt.axvline(
                    x=n_factors, 
                    color=color, 
                    linestyle="--", 
                    label=f"{method.capitalize()}: {n_factors} factors"
                )
                plt.scatter(
                    [n_factors], 
                    [variance[idx]], 
                    marker="*", 
                    s=200, 
                    color=color, 
                    zorder=5
                )
    
    # Highlight user-specified factors if provided
    if user_factors and user_factors in factors:
        idx = factors.index(user_factors)
        plt.axvline(
            x=user_factors, 
            color="orange", 
            linestyle="--", 
            label=f"User-specified: {user_factors} factors"
        )
        plt.scatter(
            [user_factors], 
            [variance[idx]], 
            marker="*", 
            s=200, 
            color="orange", 
            zorder=5
        )
    
    plt.xlabel("Number of factors")
    plt.ylabel("Total variance explained")
    plt.title("MOFA performance vs. number of factors")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def visualize_marginal_improvement(
    metrics: Dict[int, Dict[str, Any]],
    optimal_factors: Optional[Dict[str, int]] = None,
    user_factors: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a plot of marginal variance improvement by number of factors.
    
    Parameters
    ----------
    metrics : Dict[int, Dict[str, Any]]
        Dictionary with variance metrics for each factor value
    optimal_factors : Optional[Dict[str, int]]
        Dictionary mapping method names to optimal factor values, by default None
    user_factors : Optional[int]
        User-specified number of factors to highlight, by default None
    figsize : Tuple[int, int]
        Figure size, by default (12, 8)
    save_path : Optional[str]
        Path to save the figure, by default None
        
    Returns
    -------
    plt.Figure
        Created figure
        
    Examples
    --------
    >>> # Visualize marginal improvement with optimal factors highlighted
    >>> fig = visualize_marginal_improvement(metrics, optimal_factors, user_factors=15)
    >>> plt.show()
    """
    # Extract factors and variance values
    factors = sorted([k for k in metrics.keys() if metrics[k]["total_variance"] is not None])
    
    if not factors:
        print("No valid variance metrics available")
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "No valid variance metrics available", 
                 horizontalalignment="center", verticalalignment="center")
        return fig
    
    if len(factors) <= 1:
        print("Need at least two factor values to calculate marginal improvement")
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "Need at least two factor values to calculate marginal improvement", 
                 horizontalalignment="center", verticalalignment="center")
        return fig
        
    variance = [metrics[k]["total_variance"] for k in factors]
    
    # Calculate marginal improvements
    diffs = np.diff(variance)
    diffs = np.append(diffs, diffs[-1])  # Pad to match length
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Plot bars for all factors
    bars = plt.bar(factors, diffs, color="green", alpha=0.6, width=min(2, (factors[-1] - factors[0]) / len(factors) / 2))
    
    # Calculate mean improvement
    mean_improvement = np.mean(diffs)
    half_mean = mean_improvement / 2
    
    # Add threshold line
    plt.axhline(
        y=half_mean, 
        color="red", 
        linestyle="--", 
        label=f"Half of mean improvement ({half_mean:.4f})"
    )
    
    # Highlight optimal factors if provided
    colors = {"elbow": "red", "threshold": "green", "balanced": "purple"}
    highlighted_factors = set()
    
    if optimal_factors:
        for method, n_factors in optimal_factors.items():
            if n_factors in factors and n_factors not in highlighted_factors:
                highlighted_factors.add(n_factors)
                color = colors.get(method, "gray")
                idx = factors.index(n_factors)
                plt.bar(
                    [n_factors], 
                    [diffs[idx]], 
                    color=color, 
                    alpha=1.0, 
                    width=min(2, (factors[-1] - factors[0]) / len(factors) / 2),
                    label=f"{method.capitalize()}: {n_factors} factors"
                )
    
    # Highlight user-specified factors if provided
    if user_factors and user_factors in factors and user_factors not in highlighted_factors:
        idx = factors.index(user_factors)
        plt.bar(
            [user_factors], 
            [diffs[idx]], 
            color="orange", 
            alpha=1.0, 
            width=min(2, (factors[-1] - factors[0]) / len(factors) / 2),
            label=f"User-specified: {user_factors} factors"
        )
    
    plt.xlabel("Number of factors")
    plt.ylabel("Marginal variance explained")
    plt.title("Marginal improvement per additional factor")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def visualize_modality_variance(
    metrics: Dict[int, Dict[str, Any]],
    optimal_factors: Optional[Dict[str, int]] = None,
    user_factors: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a plot of variance explained per modality by number of factors.
    
    Parameters
    ----------
    metrics : Dict[int, Dict[str, Any]]
        Dictionary with variance metrics for each factor value
    optimal_factors : Optional[Dict[str, int]]
        Dictionary mapping method names to optimal factor values, by default None
    user_factors : Optional[int]
        User-specified number of factors to highlight, by default None
    figsize : Tuple[int, int]
        Figure size, by default (12, 8)
    save_path : Optional[str]
        Path to save the figure, by default None
        
    Returns
    -------
    plt.Figure
        Created figure
        
    Examples
    --------
    >>> # Visualize modality-specific variance with optimal factors highlighted
    >>> fig = visualize_modality_variance(metrics, optimal_factors, user_factors=15)
    >>> plt.show()
    """
    # Extract factors with valid metrics
    factors = sorted([k for k in metrics.keys() if metrics[k]["total_variance"] is not None])
    
    if not factors:
        print("No valid variance metrics available")
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "No valid variance metrics available", 
                 horizontalalignment="center", verticalalignment="center")
        return fig
    
    # Gather modality data
    mod_data = []
    for n_factors in factors:
        if metrics[n_factors]["modality_variance"]:
            mod_vars = metrics[n_factors]["modality_variance"]
            for mod, var in mod_vars.items():
                if var is not None:
                    mod_data.append({
                        "factors": n_factors, 
                        "modality": mod, 
                        "variance": var
                    })
    
    if not mod_data:
        print("No modality-specific variance data available")
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "No modality-specific variance data available", 
                 horizontalalignment="center", verticalalignment="center")
        return fig
    
    # Create DataFrame for plotting
    mod_df = pd.DataFrame(mod_data)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Plot modality-specific variances
    sns.lineplot(data=mod_df, x="factors", y="variance", hue="modality", marker="o", markersize=8)
    
    # Highlight optimal factors if provided
    colors = {"elbow": "red", "threshold": "green", "balanced": "purple"}
    
    if optimal_factors:
        for method, n_factors in optimal_factors.items():
            if n_factors in factors:
                color = colors.get(method, "gray")
                plt.axvline(
                    x=n_factors, 
                    color=color, 
                    linestyle="--", 
                    alpha=0.7,
                    label=f"{method.capitalize()}: {n_factors} factors"
                )
    
    # Highlight user-specified factors if provided
    if user_factors and user_factors in factors:
        plt.axvline(
            x=user_factors, 
            color="orange", 
            linestyle="--", 
            alpha=0.7,
            label=f"User-specified: {user_factors} factors"
        )
    
    plt.xlabel("Number of factors")
    plt.ylabel("Variance explained")
    plt.title("Modality-specific variance explained")
    plt.grid(True, alpha=0.3)
    
    # Adjust legend to avoid overlap
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="best", fontsize="small")
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def visualize_factor_scan_results(
    metrics: Dict[int, Dict[str, Any]],
    user_factors: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    Visualize the results of a MOFA factor scan with highlighted optimal factor selections.
    
    Parameters
    ----------
    metrics : Dict[int, Dict[str, Any]]
        Dictionary with variance metrics for each factor value
    user_factors : Optional[int]
        User-specified number of factors to highlight, by default None
    figsize : Tuple[int, int]
        Figure size for individual plots, by default (12, 8)
    save_dir : Optional[str]
        Directory to save figures, by default None
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary of created figures
        
    Examples
    --------
    >>> # Visualize factor scan results with user-specified factors
    >>> figures = visualize_factor_scan_results(metrics, user_factors=15)
    >>> # Display the figures
    >>> for name, fig in figures.items():
    ...     plt.figure(fig.number)
    ...     plt.show()
    """
    # Calculate optimal factors using different criteria
    optimal_factors = {
        "elbow": determine_optimal_factors(metrics, criterion="elbow"),
        "threshold": determine_optimal_factors(metrics, criterion="threshold", threshold=0.01),
        "balanced": determine_optimal_factors(metrics, criterion="balanced")
    }
    
    # Filter out None values
    optimal_factors = {k: v for k, v in optimal_factors.items() if v is not None}
    
    # Create save directory if provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Create figures
    figures = {}
    
    # 1. Total variance explained plot
    save_path = os.path.join(save_dir, "total_variance.png") if save_dir else None
    figures["total_variance"] = visualize_variance_explained(
        metrics, 
        optimal_factors, 
        user_factors, 
        figsize=figsize, 
        save_path=save_path
    )
    
    # 2. Marginal improvement plot
    save_path = os.path.join(save_dir, "marginal_improvement.png") if save_dir else None
    figures["marginal_improvement"] = visualize_marginal_improvement(
        metrics, 
        optimal_factors, 
        user_factors, 
        figsize=figsize, 
        save_path=save_path
    )
    
    # 3. Modality-specific variance plot
    save_path = os.path.join(save_dir, "modality_variance.png") if save_dir else None
    figures["modality_variance"] = visualize_modality_variance(
        metrics, 
        optimal_factors, 
        user_factors, 
        figsize=figsize, 
        save_path=save_path
    )
    
    # Print summary of optimal factors
    print("Optimal number of factors based on different criteria:")
    for method, n_factors in optimal_factors.items():
        print(f"  {method.capitalize()} method: {n_factors} factors")
    if user_factors:
        print(f"  User-specified: {user_factors} factors")
    
    return figures


def load_optimal_mofa_model(
    results_dict: Dict[int, Dict[str, str]],
    n_factors: int,
    mdata: Optional[md.MuData] = None,
) -> md.MuData:
    """
    Load MOFA results for a specific number of factors into a MuData object.
    
    Parameters
    ----------
    results_dict : Dict[int, Dict[str, str]]
        Dictionary with file paths for each factor value
    n_factors : int
        Number of factors to load
    mdata : Optional[md.MuData]
        MuData object to load results into, by default None
        
    Returns
    -------
    md.MuData
        MuData object with MOFA results
        
    Examples
    --------
    >>> # Load optimal MOFA model with 15 factors
    >>> optimal_mdata = load_optimal_mofa_model(factor_results, 15, mdata)
    """
    if n_factors not in results_dict:
        raise ValueError(f"No results available for {n_factors} factors")
    
    # Get file paths
    variance_file = results_dict[n_factors]["variance_file"]
    factors_file = results_dict[n_factors]["factors_file"]
    
    # Load variance data
    variance_data = dict(np.load(variance_file))
    
    # Load factor scores
    factor_scores = np.load(factors_file)
    
    # If no MuData object is provided, create a minimal one
    if mdata is None:
        # Create minimal MuData with just factor scores
        mdata = md.MuData()
        mdata.obsm["X_mofa"] = factor_scores
    else:
        # Add factor scores to existing MuData
        mdata.obsm["X_mofa"] = factor_scores
    
    # Add variance data
    if "mofa" not in mdata.uns:
        mdata.uns["mofa"] = {}
    
    mdata.uns["mofa"]["variance"] = variance_data
    mdata.uns["mofa"]["params"] = {
        "model": {"n_factors": n_factors}
    }
    
    # Return MuData with MOFA results
    return mdata


def run_complete_mofa_analysis(
    mdata: md.MuData,
    factor_range: Iterable[int] = range(5, 51, 5),
    use_layer: Optional[str] = "log2_centered",
    use_var: Optional[str] = None,
    seed: int = 42,
    models_dir: str = "mofa_models",
    overwrite: bool = False,
    user_factors: Optional[int] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a complete MOFA analysis workflow including factor scan, visualization,
    and optimal factor determination.
    
    Parameters
    ----------
    mdata : md.MuData
        Multi-modal data object
    factor_range : Iterable[int]
        Range of factors to test, by default range(5, 51, 5)
    use_layer : Optional[str]
        Layer to use for analysis, by default "log2_centered"
    use_var : Optional[str]
        Variable selection to use, by default None
    seed : int
        Random seed for reproducibility, by default 42
    models_dir : str
        Directory to store model files, by default "mofa_models"
    overwrite : bool
        Whether to overwrite existing results, by default False
    user_factors : Optional[int]
        User-specified number of factors, by default None
    save_dir : Optional[str]
        Directory to save figures, by default None
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with complete analysis results
        
    Examples
    --------
    >>> # Run complete MOFA analysis workflow
    >>> results = run_complete_mofa_analysis(
    ...     mdata,
    ...     factor_range=range(5, 51, 5),
    ...     use_layer="log2_centered",
    ...     models_dir="mofa_results",
    ...     user_factors=20,
    ...     save_dir="mofa_figures"
    ... )
    >>> 
    >>> # Access optimal models
    >>> optimal_balanced = results["optimal_models"]["balanced"]
    >>> 
    >>> # Display figures
    >>> import matplotlib.pyplot as plt
    >>> for name, fig in results["figures"].items():
    ...     plt.figure(fig.number)
    ...     plt.show()
    """
    print(f"[{_get_timestamp()}] Starting MOFA analysis workflow...")
    
    # Step 1: Run MOFA factor scan
    print(f"[{_get_timestamp()}] Running MOFA factor scan...")
    factor_results = run_mofa_factor_scan(
        mdata,
        factor_range=factor_range,
        use_layer=use_layer,
        use_var=use_var,
        seed=seed,
        models_dir=models_dir,
        overwrite=overwrite
    )
    
    # Step 2: Calculate variance metrics
    print(f"[{_get_timestamp()}] Calculating variance metrics...")
    metrics = calculate_variance_metrics(factor_results)
    
    # Step 3: Determine optimal number of factors
    print(f"[{_get_timestamp()}] Determining optimal number of factors...")
    optimal_factors = {
        "elbow": determine_optimal_factors(metrics, criterion="elbow"),
        "threshold": determine_optimal_factors(metrics, criterion="threshold", threshold=0.01),
        "balanced": determine_optimal_factors(metrics, criterion="balanced")
    }
    
    # Filter out None values
    optimal_factors = {k: v for k, v in optimal_factors.items() if v is not None}
    
    # Step 4: Visualize factor scan results
    print(f"[{_get_timestamp()}] Visualizing factor scan results...")
    figures = visualize_factor_scan_results(
        metrics,
        user_factors=user_factors,
        figsize=(12, 8),
        save_dir=save_dir
    )
    
    # Step 5: Load optimal models
    print(f"[{_get_timestamp()}] Loading optimal models...")
    optimal_models = {}
    
    for method, n_factors in optimal_factors.items():
        try:
            optimal_models[method] = load_optimal_mofa_model(
                factor_results,
                n_factors,
                mdata=None  # Don't load into original MuData
            )
            print(f"Loaded optimal model for {method} method: {n_factors} factors")
        except Exception as e:
            print(f"Error loading optimal model for {method} method: {str(e)}")
    
    # Load user-specified model if provided
    if user_factors:
        try:
            optimal_models["user"] = load_optimal_mofa_model(
                factor_results,
                user_factors,
                mdata=None  # Don't load into original MuData
            )
            print(f"Loaded user-specified model: {user_factors} factors")
        except Exception as e:
            print(f"Error loading user-specified model: {str(e)}")
    
    print(f"[{_get_timestamp()}] MOFA analysis workflow completed.")
    
    return {
        "factor_results": factor_results,
        "metrics": metrics,
        "optimal_factors": optimal_factors,
        "figures": figures,
        "optimal_models": optimal_models
    }


def plot_mofa_factor_histograms(
    mdata,
    factors: Optional[Union[int, List[int]]] = None,
    n_bins: int = 30,
    n_cols: int = 3,
    figsize: Tuple[float, float] = None,
    kde: bool = True,
    color: str = 'steelblue',
    return_fig: bool = False
):
    """
    Plot histograms of MOFA factor values.
    
    Parameters
    ----------
    mdata : MuData
        MuData object with MOFA results
    factors : int or list of ints, optional
        Factors to plot. If None, all factors are plotted.
    n_bins : int, default=30
        Number of bins for histograms
    n_cols : int, default=3
        Number of columns in the grid plot
    figsize : tuple, optional
        Figure size
    kde : bool, default=True
        Whether to overlay a kernel density estimate
    color : str, default='steelblue'
        Color for the histogram bars
    return_fig : bool, default=False
        Whether to return the figure object
    
    Returns
    -------
    matplotlib.figure.Figure, optional
        Figure object, if return_fig is True
    """
    # Extract MOFA factors
    X_mofa = mdata.obsm["X_mofa"]
    
    # Get number of factors
    n_factors = X_mofa.shape[1]
    
    # Select factors to plot
    if factors is None:
        factors = list(range(n_factors))
    elif isinstance(factors, int):
        factors = [factors]
    
    # Create a DataFrame with factor values
    df = pd.DataFrame(
        X_mofa[:, factors],
        index=mdata.obs_names,
        columns=[f"Factor {i+1}" for i in factors]
    )
    
    # Calculate number of rows needed
    n_factors_to_plot = len(factors)
    n_rows = int(np.ceil(n_factors_to_plot / n_cols))
    
    # Set up the figure
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single subplot case
    if n_factors_to_plot == 1:
        axes = np.array([axes])
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
    
    # Plot each factor
    for i, factor_idx in enumerate(factors):
        if i < len(axes):
            ax = axes[i]
            factor_name = f"Factor {factor_idx+1}"
            
            # Get factor values
            factor_values = df[factor_name]
            
            # Plot histogram
            sns.histplot(
                factor_values,
                bins=n_bins,
                kde=kde,
                color=color,
                ax=ax
            )
            
            # Set labels and title
            ax.set_xlabel("Factor Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {factor_name}")
            
            # Add vertical line at mean
            mean_val = factor_values.mean()
            ax.axvline(mean_val, color='red', linestyle='--', 
                       label=f'Mean: {mean_val:.2f}')
            
            # Add vertical line at median
            median_val = factor_values.median()
            ax.axvline(median_val, color='green', linestyle=':', 
                       label=f'Median: {median_val:.2f}')
            
            # Add legend
            ax.legend(fontsize='small')
    
    # Remove any unused subplots
    for i in range(n_factors_to_plot, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()


def regress_factors_with_formula(
    mdata: MuData,
    formula: str,
    factors: Optional[Union[int, List[int]]] = None,
    modality: str = "transcriptomics"  # Default modality
) -> pd.DataFrame:
    """
    Apply formula-based regression to MOFA factors.
    
    Parameters
    ----------
    mdata : MuData
        MuData object with MOFA results in obsm['X_mofa']
    formula : str
        Formula for regression in patsy format (e.g., '~ case + gender').
        Do not include the dependent variable as it will be each factor.
    factors : int or list of ints, optional
        Factors to analyze. If None, all factors are used.
    modality : str, default="transcriptomics"
        Modality to use for observation variables.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with regression statistics for each factor.
    """
    # Check if MOFA results exist
    if 'X_mofa' not in mdata.obsm:
        raise ValueError("MOFA results not found in mdata.obsm['X_mofa']")
    
    # Check if modality exists
    if modality not in mdata.mod:
        raise ValueError(f"Modality '{modality}' not found in MuData object. Available modalities: {list(mdata.mod.keys())}")
    
    # Extract MOFA factors
    X_mofa = mdata.obsm["X_mofa"]
    
    # Get number of factors
    n_factors = X_mofa.shape[1]
    
    # Select factors to analyze
    if factors is None:
        factors = list(range(n_factors))
    elif isinstance(factors, int):
        factors = [factors]
    
    # Validate formula
    if not formula.startswith('~'):
        raise ValueError("Formula must start with '~'")
    
    # Create data dictionary first, then create DataFrame all at once
    data_dict = {}
    
    # Add observation variables from the specified modality
    for col in mdata[modality].obs.columns:
        data_dict[col] = mdata[modality].obs[col].values
    
    # Add factors to data dictionary
    for factor_idx in factors:
        data_dict[f"Factor_{factor_idx+1}"] = X_mofa[:, factor_idx]
    
    # Create DataFrame all at once to avoid fragmentation warnings
    df = pd.DataFrame(data_dict, index=mdata.obs_names)
    
    # Initialize results list
    results_list = []
    
    # Process each factor
    for factor_idx in factors:
        factor_name = f"Factor_{factor_idx+1}"
        
        # Build formula with current factor as dependent variable
        full_formula = f"{factor_name} {formula}"
        
        try:
            # Fit the model using statsmodels formula API
            model = smf.ols(formula=full_formula, data=df)
            result = model.fit()
            
            # Extract results for each term
            # Skip the intercept (index 0)
            for i, term in enumerate(result.params.index[1:], 1):
                # Extract confidence intervals
                conf_ints = result.conf_int()
                conf_lower = conf_ints.iloc[i, 0]
                conf_upper = conf_ints.iloc[i, 1]
                
                # Store result for this term
                results_list.append({
                    "factor_idx": factor_idx,  # Store numeric index
                    "factor_name": factor_name,
                    "term": term,
                    "estimate": result.params[i],
                    "std_err": result.bse[i],
                    "statistic": result.tvalues[i],
                    "p_value": result.pvalues[i],
                    "conf_int_lower": conf_lower,
                    "conf_int_upper": conf_upper,
                    "nobs": result.nobs,
                    "rsquared": result.rsquared,
                    "rsquared_adj": result.rsquared_adj
                })
                
        except Exception as e:
            print(f"Error processing factor {factor_name}: {str(e)}")
    
    # Handle empty results
    if not results_list:
        print("No regression results were generated. Check your data and formula.")
        return pd.DataFrame()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Add FDR correction (Benjamini-Hochberg)
    if not results_df.empty and "p_value" in results_df.columns:
        # Group by term and calculate FDR-corrected p-values
        terms = results_df["term"].unique()
        q_values_list = []
        significant_list = []
        
        # Store indices for each term to use in assignment
        term_indices = {}
        
        for term in terms:
            mask = results_df["term"] == term
            # Store indices for this term
            term_indices[term] = results_df[mask].index
            p_values = results_df.loc[mask, "p_value"].values
            
            if len(p_values) > 0:
                try:
                    # Apply multiple testing correction
                    reject, q_values, _, _ = multipletests(
                        p_values, method="fdr_bh"
                    )
                    
                    # Store results for later assignment
                    for idx, q_value, reject_val in zip(term_indices[term], q_values, reject):
                        q_values_list.append((idx, q_value))
                        significant_list.append((idx, reject_val))
                        
                except Exception as e:
                    print(f"Error applying multiple testing correction for {term}: {str(e)}")
                    # Create NaN/False entries for this term
                    for idx in term_indices[term]:
                        q_values_list.append((idx, np.nan))
                        significant_list.append((idx, False))
        
        # Add columns efficiently using values
        if q_values_list:
            # Create new columns
            results_df["q_value"] = np.nan
            results_df["significant"] = False
            
            # Assign values
            for idx, q_value in q_values_list:
                results_df.at[idx, "q_value"] = q_value
                
            for idx, sig_value in significant_list:
                results_df.at[idx, "significant"] = sig_value
    
    print(f"Completed regression analysis for {len(results_df)} factor-term pairs.")
    return results_df


def summarize_factor_regression(
    regression_results: pd.DataFrame,
    alpha: float = 0.05,
    group_by_factor: bool = False
) -> pd.DataFrame:
    """
    Create a summary table of regression results.
    
    Parameters
    ----------
    regression_results : pd.DataFrame
        DataFrame with regression results from regress_factors_with_formula()
    alpha : float, default=0.05
        Significance threshold for q-values
    group_by_factor : bool, default=False
        If True, organize results by factor; otherwise by term
        
    Returns
    -------
    pd.DataFrame
        Formatted summary table
    """
    if regression_results.empty:
        print("No regression results to summarize.")
        return pd.DataFrame()
    
    # Use q_value if available, otherwise use p_value
    p_col = "q_value" if "q_value" in regression_results.columns else "p_value"
    
    # Create significance mask
    results_df = regression_results.copy()
    results_df['significant'] = results_df[p_col] < alpha
    
    # Filter for significant results
    sig_results = results_df[results_df['significant']].copy()
    
    if sig_results.empty:
        print(f"No significant associations found at alpha = {alpha}")
        return pd.DataFrame()
    
    # Select relevant columns
    summary_cols = [
        'factor_name', 'term', 'estimate', 'std_err',
        'p_value', 'q_value' if 'q_value' in results_df.columns else 'p_value',
        'rsquared', 'nobs'
    ]
    
    # Check for excluded columns
    available_cols = []
    for col in summary_cols:
        if col in sig_results.columns or (col == 'q_value' and 'q_value' not in sig_results.columns):
            if col != 'q_value' or 'q_value' in sig_results.columns:
                available_cols.append(col)
    
    summary_df = sig_results[available_cols].copy()
    
    # Group by factor or term depending on preference
    # Always sort by p-value and q-value (if available), then by factor or term
    sort_cols = []
    if 'p_value' in summary_df.columns:
        sort_cols.append('p_value')
    if 'q_value' in summary_df.columns:
        sort_cols.append('q_value')
    # Add grouping column
    if group_by_factor:
        sort_cols = ['factor_name'] + sort_cols
    else:
        sort_cols = ['term'] + sort_cols
    summary_df = summary_df.sort_values(sort_cols, ascending=True)

    # Format numeric columns (after sorting)
    for col in ['estimate', 'std_err']:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].map('{:.3f}'.format)

    for col in ['p_value', 'q_value']:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].map('{:.2e}'.format)

    for col in ['rsquared']:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].map('{:.3f}'.format)
    
    return summary_df


def factor_term_scatterplot(
    mdata: MuData,
    factor: int,
    formula_name: str,
    term: Optional[str] = None
) -> None:
    """
    Scatterplot of a MOFA factor against a sample attribute, with regression line.

    Parameters
    ----------
    mdata : MuData
        MuData object containing MOFA results and sample attributes.
    factor : int
        Factor index or name to plot.
    formula_name : str
        Name of the regression formula (used as key in mdata.uns).
    term : str or None, optional
        Name of the sample attribute to plot on the x-axis. If None, uses formula_name.

    Returns
    -------
    None
    """
    # If term is None, use formula_name as the term
    if term is None:
        term = formula_name
    df = _factor_term_sample_attrs(mdata, factor, formula_name, term)
    # Get the actual factor name for labeling
    factor_regression_key = FACTOR_REGRESSION_STR.format(formula_name)
    factor_term_summaries = mdata.uns[factor_regression_key]
    if (factor_term_summaries["factor_name"] == factor).any():
        factor_name = factor
    elif (factor_term_summaries["factor_idx"] == factor).any():
        factor_name = factor_term_summaries[factor_term_summaries["factor_idx"] == factor]["factor_name"].iloc[0]
    else:
        factor_name = str(factor)
    # Plot the data with regression line
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(7, 5))
    sns.regplot(data=df, x="sample_attribute", y="factor_values", scatter_kws={'alpha':0.7})
    plt.xlabel(term)
    plt.ylabel(factor_name)
    # Only show (formula_name) in title if term != formula_name
    if term == formula_name:
        title = f"{factor_name} vs {term}"
    else:
        title = f"{factor_name} vs {term} ({formula_name})"
    plt.title(title)
    plt.tight_layout()
    plt.show()


def _factor_term_sample_attrs(
    mdata: MuData,
    factor: int,
    formula_name: str,
    term: str
) -> pd.DataFrame:
    """
    Helper to extract a DataFrame of sample attribute and factor values for plotting.

    Parameters
    ----------
    mdata : MuData
        MuData object containing MOFA results and sample attributes.
    factor : int
        Factor index or name to extract.
    formula_name : str
        Name of the regression formula (used as key in mdata.uns).
    term : str
        Name of the sample attribute to extract.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'sample_attribute' and 'factor_values'.
    """
    factor_regression_key = FACTOR_REGRESSION_STR.format(formula_name)
    # validate that the key exists
    if factor_regression_key not in mdata.uns:
        raise KeyError(f"No regression results found in mdata.uns under key '{factor_regression_key}'")

    factor_term_summaries = mdata.uns[factor_regression_key]
    # see if factor_name exists or factor_idx was provided in the factor arg
    if (factor_term_summaries["factor_name"] == factor).any():
        factor_df = factor_term_summaries[factor_term_summaries["factor_name"] == factor]
    elif (factor_term_summaries["factor_idx"] == factor).any():
        factor_df = factor_term_summaries[factor_term_summaries["factor_idx"] == factor]
    else:
        raise ValueError(f"Factor '{factor}' did not match the factor_name or factor_idx values factor_term_summaries")

    factor_idx = factor_df["factor_idx"].iloc[0]

    # lookup factor values from obsm
    factor_values = mdata.obsm["X_mofa"][:, factor_idx]

    # pull out the relevant sample attribute
    sample_attribute = mdata[mdata.mod_names[0]].obs

    # check for the term in the sample attribute
    if term in sample_attribute.columns:
        sample_attribute = sample_attribute[term]
    else:
        raise ValueError(f"Term '{term}' not found in sample attribute")

    df = pd.DataFrame({"sample_attribute": sample_attribute, "factor_values": factor_values})

    return df


# Additional utility functions with underscore prefix

def _get_timestamp() -> str:
    """Get current timestamp formatted as string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _calculate_per_factor_variance(variance_data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Calculate per-factor variance summed across modalities."""
    per_factor_variance = None
    
    for mod, var_array in variance_data.items():
        if per_factor_variance is None:
            per_factor_variance = var_array.copy()
        else:
            per_factor_variance += var_array
            
    return per_factor_variance


def _determine_optimal_elbow(
    factors: List[int], 
    variance: List[float], 
    diffs: np.ndarray
) -> int:
    """Determine optimal factors using the elbow method."""
    # Elbow method: find where the rate of change of the differences is greatest
    if len(diffs) <= 1:
        return factors[0]

    # Calculate the acceleration (approximating second derivative)
    acceleration = np.diff(diffs)
    acceleration = np.append(acceleration, acceleration[-1])  # Pad to match length

    # Find the point of maximum acceleration
    elbow_idx = np.argmax(np.abs(acceleration)) + 1  # +1 because we're working with diffs

    return factors[elbow_idx]


def _determine_optimal_threshold(
    factors: List[int], 
    diffs: np.ndarray, 
    threshold: float
) -> int:
    """Determine optimal factors using the threshold method."""
    # Threshold method: find first point where marginal improvement falls below threshold
    for i, diff in enumerate(diffs):
        if diff < threshold:
            return factors[i]

    # If all improvements are above threshold, return the maximum
    return factors[-1]


def _determine_optimal_balanced(
    factors: List[int], 
    variance: List[float]
) -> int:
    """Determine optimal factors using a balanced approach."""
    # Balanced approach: consider both variance explained and number of factors
    # Normalize factors and variance to [0, 1]
    norm_factors = (np.array(factors) - min(factors)) / (max(factors) - min(factors))
    norm_variance = (np.array(variance) - min(variance)) / (max(variance) - min(variance))

    # Calculate a score balancing parsimony and variance explained
    # Higher variance is better, fewer factors is better
    scores = norm_variance - 0.5 * norm_factors

    # Find the factor value with the highest score
    best_idx = np.argmax(scores)

    return factors[best_idx]
