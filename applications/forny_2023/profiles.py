"""
This module contains functions for regression and factor analysis.
"""

from joblib import Parallel, delayed
import os
from typing import List, Optional, Tuple, Dict, Union, Any

from anndata import AnnData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mudata import MuData
import seaborn as sns
import scipy.sparse as sp
from scipy.stats import kstest
import statsmodels.api as sm
import statsmodels.formula.api as smf


def apply_regression_per_feature(
    adata: AnnData,
    formula: str,
    layer: Optional[str] = None,
    n_jobs: int = -1,
    batch_size: int = 100,
    progress_bar: bool = True
) -> pd.DataFrame:
    """
    Apply linear regression to each feature in an AnnData object and return statistics.
    This simplified version only supports direct regression against specified factors
    without additional covariates from obs.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The annotated data matrix.
    formula : str
        Formula for regression in patsy format (e.g., '~ batch').
        Don't include the dependent variable, as each feature will be used.
    layer : Optional[str], optional
        If provided, use this layer instead of X. The layer can be a string referring
        to a layer in adata.layers, or a key in adata.obsm for alternative feature matrices.
    n_jobs : int, optional
        Number of parallel jobs. -1 means using all processors.
    batch_size : int, optional
        Number of features to process in each batch.
    progress_bar : bool, optional
        Whether to display a progress bar.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with regression statistics for each feature.
    """
    # Get feature names based on data source
    if layer is None:
        feature_names = adata.var_names
    elif layer in adata.layers:
        feature_names = adata.var_names
    elif layer in adata.obsm:
        # For obsm matrices, we need to generate feature names
        if hasattr(adata.obsm[layer], 'columns'):
            # If it's a DataFrame with column names
            feature_names = adata.obsm[layer].columns
        else:
            # Otherwise, generate numeric feature names
            n_features = adata.obsm[layer].shape[1]
            feature_names = [f"{layer}_feature_{i}" for i in range(n_features)]
    else:
        raise ValueError(f"Layer '{layer}' not found in adata.layers or adata.obsm")
    
    # Get data to regress
    if layer is None:
        if isinstance(adata.X, np.ndarray):
            X_data = adata.X
        else:
            X_data = adata.X.toarray()
    elif layer in adata.layers:
        if isinstance(adata.layers[layer], np.ndarray):
            X_data = adata.layers[layer]
        else:
            X_data = adata.layers[layer].toarray()
    elif layer in adata.obsm:
        if isinstance(adata.obsm[layer], pd.DataFrame):
            X_data = adata.obsm[layer].values
        elif isinstance(adata.obsm[layer], np.ndarray):
            X_data = adata.obsm[layer]
        elif sp.issparse(adata.obsm[layer]):
            X_data = adata.obsm[layer].toarray()
        else:
            raise ValueError(f"Unsupported data type for obsm['{layer}']")
    
    # Parse variables from formula
    # Formula should be in the format "~ var1 + var2 + ..."
    if not formula.startswith('~'):
        raise ValueError("Formula must start with '~'")
    
    formula_vars = [var.strip() for var in formula[1:].split('+')]
    formula_vars = [var for var in formula_vars if var]  # Remove empty strings
    
    if not formula_vars:
        raise ValueError("No variables found in formula")
    
    # Extract variables from adata.obs
    formula_data = {}
    for var in formula_vars:
        if var not in adata.obs.columns:
            raise ValueError(f"Variable '{var}' not found in adata.obs")
        formula_data[var] = adata.obs[var].values
    
    # Define a function to process a single feature
    def process_feature(i: int) -> List[Dict[str, Any]]:
        feature_name = feature_names[i]
        y = X_data[:, i]
        
        # Skip features with zero variance
        if np.var(y) == 0:
            return []
        
        feature_results: List[Dict[str, Any]] = []
        
        try:
            # Create design matrix manually for each variable in formula
            for var_name in formula_vars:
                var_values = formula_data[var_name]
                
                # For categorical variables, create dummy variables
                if pd.api.types.is_categorical_dtype(adata.obs[var_name]):
                    categories = adata.obs[var_name].cat.categories
                    
                    # Skip if there's only one category
                    if len(categories) <= 1:
                        continue
                    
                    # For each category, perform a separate regression
                    for cat in categories[1:]:  # Skip first category as reference
                        # Create dummy variable (1 if category matches, 0 otherwise)
                        dummy = (var_values == cat).astype(float)
                        
                        # Skip if dummy has no variation
                        if np.var(dummy) == 0:
                            continue
                        
                        # Simple linear regression: y = b0 + b1*x
                        X = sm.add_constant(dummy)
                        model = sm.OLS(y, X)
                        result = model.fit()
                        
                        # Extract confidence intervals - FIX: access by index, not .loc
                        conf_ints = result.conf_int()
                        conf_lower = conf_ints[0][1]  # [0][1] for lower bound of coefficient
                        conf_upper = conf_ints[1][1]  # [1][1] for upper bound of coefficient
                        
                        # Store results only for the coefficient (not intercept)
                        coef_name = f"{var_name}_{cat}"
                        feature_results.append(
                            {
                                "feature": feature_name,
                                "coefficient": coef_name,
                                "estimate": result.params[1],  # Skip intercept
                                "std_err": result.bse[1],
                                "statistic": result.tvalues[1],
                                "p_value": result.pvalues[1],
                                "conf_int_lower": conf_lower,
                                "conf_int_upper": conf_upper,
                            }
                        )
                # For continuous variables, perform simple regression
                else:
                    # Simple linear regression: y = b0 + b1*x
                    X = sm.add_constant(var_values)
                    model = sm.OLS(y, X)
                    result = model.fit()
                    
                    # Extract confidence intervals - FIX: access by index, not .loc
                    conf_ints = result.conf_int()
                    conf_lower = conf_ints[0][1]  # [0][1] for lower bound of coefficient
                    conf_upper = conf_ints[1][1]  # [1][1] for upper bound of coefficient
                    
                    # Store results only for the coefficient (not intercept)
                    feature_results.append(
                        {
                            "feature": feature_name,
                            "coefficient": var_name,
                            "estimate": result.params[1],  # Skip intercept
                            "std_err": result.bse[1],
                            "statistic": result.tvalues[1],
                            "p_value": result.pvalues[1],
                            "conf_int_lower": conf_lower,
                            "conf_int_upper": conf_upper,
                        }
                    )
        except Exception as e:
            print(f"Error in feature {feature_name}: {str(e)}")
        
        return feature_results
    
    # Create feature indices
    feature_indices: List[int] = list(range(X_data.shape[1]))
    
    # Verbose setting for progress bar
    verbose = 10 if progress_bar else 0
    
    # Run parallel processing with joblib
    print(f"Starting regression analysis with {n_jobs} cores...")
    results_nested: List[List[Dict[str, Any]]] = Parallel(n_jobs=n_jobs, batch_size=batch_size, verbose=verbose)(
        delayed(process_feature)(i) for i in feature_indices
    )
    
    # Flatten results
    results: List[Dict[str, Any]] = []
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
    adata: AnnData,
    results_df: pd.DataFrame,
    key_added: str = "regression_results",
    fdr_cutoff: float = 0.05,
    inplace: bool = True,
    effect_only_for_significant: bool = False,  # New parameter
    add_all_stats: bool = True  # Add all statistics columns
) -> Optional[AnnData]:
    """
    Add regression results dataframe to an AnnData object in a structured way.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The annotated data matrix.
    results_df : pd.DataFrame
        DataFrame with regression results as produced by apply_regression_per_feature().
    key_added : str, optional
        Key under which to add the results in adata.uns.
    fdr_cutoff : float, optional
        Significance threshold for FDR-corrected p-values. Default is 0.05.
    inplace : bool, optional
        If True, modify adata inplace. Otherwise return a copy. Default is True.
    effect_only_for_significant : bool, optional
        If True, effect sizes are only stored for significant features.
        If False, effect sizes are stored for all features. Default is False.
    add_all_stats : bool, optional
        If True, add all available statistics (p-values, q-values, t-statistics, std errors)
        to the adata.var dataframe. Default is True.
        
    Returns
    -------
    Optional[anndata.AnnData]
        Depends on `inplace` parameter. If `inplace=True`, returns None,
        otherwise returns a modified copy of the AnnData object.
    """
    if not inplace:
        adata = adata.copy()
    
    # Make a copy to avoid modifying the original results
    results = results_df.copy()
    
    # Create a hierarchical dictionary to store in adata.uns
    regression_dict: Dict[str, Any] = {
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
    # Create a significance mask for each coefficient
    for coef in results["coefficient"].unique():
        coef_safe = coef.replace(" ", "_").replace("-", "_")
        
        # Determine the p-value column to use (FDR-corrected if available)
        p_col = "fdr_bh" if "fdr_bh" in results.columns else "p_value"
        
        # Get significant features for this coefficient
        sig_mask = (results["coefficient"] == coef) & (results[p_col] < fdr_cutoff)
        sig_features = results.loc[sig_mask, "feature"].unique()
        
        # Create a binary indicator in adata.var
        adata.var[f"sig_{coef_safe}"] = adata.var_names.isin(sig_features)
        
        # Filter for the current coefficient
        coef_results = results[results["coefficient"] == coef]
        
        # Create effect size dictionary for all features
        effect_dict: Dict[str, float] = {}
        for _, row in coef_results.iterrows():
            effect_dict[row["feature"]] = row["estimate"]
        
        # Create effect size column for all features or only significant ones
        if effect_only_for_significant:
            # Original behavior: add effect sizes only for significant features
            adata.var[f"effect_{coef_safe}"] = pd.Series(
                [
                    effect_dict.get(f, np.nan) if f in sig_features else np.nan
                    for f in adata.var_names
                ],
                index=adata.var_names,
            )
        else:
            # New behavior: add effect sizes for all features
            adata.var[f"effect_{coef_safe}"] = pd.Series(
                [effect_dict.get(f, np.nan) for f in adata.var_names],
                index=adata.var_names,
            )
        
        # Add all statistics as separate columns for all features
        if add_all_stats:
            # Raw p-values
            pval_dict: Dict[str, float] = {}
            for _, row in coef_results.iterrows():
                pval_dict[row["feature"]] = row["p_value"]
                
            adata.var[f"pval_{coef_safe}"] = pd.Series(
                [pval_dict.get(f, np.nan) for f in adata.var_names], 
                index=adata.var_names
            )
            
            # FDR-corrected p-values (q-values)
            if "fdr_bh" in coef_results.columns:
                qval_dict: Dict[str, float] = {}
                for _, row in coef_results.iterrows():
                    qval_dict[row["feature"]] = row["fdr_bh"]
                    
                adata.var[f"qval_{coef_safe}"] = pd.Series(
                    [qval_dict.get(f, np.nan) for f in adata.var_names], 
                    index=adata.var_names
                )
            
            # T-statistics
            tstat_dict: Dict[str, float] = {}
            for _, row in coef_results.iterrows():
                tstat_dict[row["feature"]] = row["statistic"]
                
            adata.var[f"tstat_{coef_safe}"] = pd.Series(
                [tstat_dict.get(f, np.nan) for f in adata.var_names], 
                index=adata.var_names
            )
            
            # Standard errors
            stderr_dict: Dict[str, float] = {}
            for _, row in coef_results.iterrows():
                stderr_dict[row["feature"]] = row["std_err"]
                
            adata.var[f"stderr_{coef_safe}"] = pd.Series(
                [stderr_dict.get(f, np.nan) for f in adata.var_names], 
                index=adata.var_names
            )
        else:
            # Just add the selected p-value column as before
            pval_dict: Dict[str, float] = {}
            for _, row in coef_results.iterrows():
                pval_dict[row["feature"]] = row[p_col]
                
            adata.var[f"pval_{coef_safe}"] = pd.Series(
                [pval_dict.get(f, np.nan) for f in adata.var_names], 
                index=adata.var_names
            )
    
    # Add a summarized view for quick access to top features per coefficient
    top_features: Dict[str, List[Dict[str, Any]]] = {}
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


def plot_pvalue_histograms(
    data: Union[AnnData, MuData],
    regression_key: str = "regression_results",
    output_dir: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    include_stats: bool = True,
    show_ks_test: bool = True,
    fdr_cutoff: float = 0.05
) -> Dict[str, Dict[str, plt.Figure]]:
    """
    Generate histograms of p-values and FDR-corrected p-values for all coefficients 
    in regression results stored in AnnData or MuData objects.
    
    Parameters
    ----------
    data : Union[anndata.AnnData, mudata.MuData]
        The annotated data matrix or multi-modal data object containing regression results.
    regression_key : str, optional
        Key under which regression results are stored in .uns. Default is "regression_results".
    output_dir : Optional[str], optional
        Directory to save histogram plots. If None, plots are displayed but not saved.
    figsize : Tuple[int, int], optional
        Figure size (width, height) in inches. Default is (12, 8).
    include_stats : bool, optional
        Whether to include summary statistics on the plots. Default is True.
    show_ks_test : bool, optional
        Whether to show Kolmogorov-Smirnov test results comparing p-value distribution
        to the uniform distribution. Default is True.
    fdr_cutoff : float, optional
        Significance threshold for FDR-corrected p-values. Default is 0.05.
        
    Returns
    -------
    Dict[str, Dict[str, plt.Figure]]
        Dictionary of generated figures indexed by modality and coefficient.
    """
    # Set plot style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Initialize dictionary to store figure objects
    figures: Dict[str, Dict[str, plt.Figure]] = {}
    
    # Function to process a single AnnData object
    def process_anndata(
        adata: AnnData, 
        modality_name: Optional[str] = None
    ) -> Dict[str, plt.Figure]:
        # Check if regression results exist
        if regression_key not in adata.uns:
            print(f"No regression results found under key '{regression_key}'"
                  f"{' in modality ' + modality_name if modality_name else ''}.")
            return {}
        
        # Get regression results
        regression_results = adata.uns[regression_key]
        
        # Convert results to DataFrame if needed
        if isinstance(regression_results, dict) and "results" in regression_results:
            results_df = pd.DataFrame(regression_results["results"])
        elif isinstance(regression_results, pd.DataFrame):
            results_df = regression_results
        else:
            print(f"Unexpected format for regression results" 
                  f"{' in modality ' + modality_name if modality_name else ''}.")
            return {}
        
        mod_figures: Dict[str, plt.Figure] = {}
        
        # Get unique coefficients
        coefficients: List[str] = results_df["coefficient"].unique().tolist()
        
        for coef in coefficients:
            # Filter results for current coefficient
            coef_results = results_df[results_df["coefficient"] == coef]
            
            # Create figure - now with 2-3 panels depending on available data
            n_panels = 3 if "fdr_bh" in coef_results.columns else 2
            fig, axes = plt.subplots(1, n_panels, figsize=(figsize[0]*n_panels/2, figsize[1]))
            fig.suptitle(f"P-value Distribution for Coefficient: {coef}"
                         f"{' in ' + modality_name if modality_name else ''}", 
                         fontsize=16)
            
            # Raw p-values
            sns.histplot(coef_results["p_value"], bins=20, kde=True, ax=axes[0])
            axes[0].set_title("Raw P-values")
            axes[0].set_xlabel("P-value")
            axes[0].set_ylabel("Count")
            
            # Add reference uniform distribution line
            x = np.linspace(0, 1, 100)
            y = len(coef_results) * 0.05  # Adjust height based on data
            axes[0].plot(x, [y] * 100, 'r--', label='Uniform Distribution')
            axes[0].legend()
            
            # Add summary statistics
            if include_stats:
                avg_p = coef_results["p_value"].mean()
                median_p = coef_results["p_value"].median()
                sig_count = sum(coef_results["p_value"] < 0.05)
                sig_pct = 100 * sig_count / len(coef_results)
                
                stats_text = f"Mean: {avg_p:.4f}\nMedian: {median_p:.4f}\n"
                stats_text += f"Significant: {sig_count}/{len(coef_results)} ({sig_pct:.1f}%)"
                
                axes[0].text(0.05, 0.95, stats_text, 
                             transform=axes[0].transAxes, 
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # QQ plot of p-values (to check for uniform distribution)
            # This helps identify deviations from expected p-value distribution
            pvals = np.sort(coef_results["p_value"])
            expected = np.linspace(0, 1, len(pvals) + 1)[1:]
            
            axes[1].scatter(expected, pvals, alpha=0.5)
            axes[1].plot([0, 1], [0, 1], 'r--')
            axes[1].set_title("P-value QQ Plot")
            axes[1].set_xlabel("Expected P-value")
            axes[1].set_ylabel("Observed P-value")
            
            # FDR-corrected p-values (if available)
            if "fdr_bh" in coef_results.columns:
                sns.histplot(coef_results["fdr_bh"], bins=20, kde=True, ax=axes[2])
                axes[2].set_title("FDR-corrected P-values (Benjamini-Hochberg)")
                axes[2].set_xlabel("Adjusted P-value")
                axes[2].set_ylabel("Count")
                
                # Add reference uniform distribution line
                axes[2].plot(x, [y] * 100, 'r--', label='Uniform Distribution')
                axes[2].legend()
                
                # Add summary statistics
                if include_stats:
                    avg_fdr = coef_results["fdr_bh"].mean()
                    median_fdr = coef_results["fdr_bh"].median()
                    sig_count_fdr = sum(coef_results["fdr_bh"] < fdr_cutoff)
                    sig_pct_fdr = 100 * sig_count_fdr / len(coef_results)
                    
                    stats_text = f"Mean: {avg_fdr:.4f}\nMedian: {median_fdr:.4f}\n"
                    stats_text += f"Significant: {sig_count_fdr}/{len(coef_results)} ({sig_pct_fdr:.1f}%)"
                    
                    axes[2].text(0.05, 0.95, stats_text, 
                                 transform=axes[2].transAxes, 
                                 verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Calculate KS test against uniform distribution (expected for null hypothesis)
            if show_ks_test:
                ks_stat, ks_pval = kstest(coef_results["p_value"], 'uniform')
                
                # Add KS test result
                fig.text(0.5, 0.01, 
                         f"Kolmogorov-Smirnov test against uniform distribution: "
                         f"statistic={ks_stat:.4f}, p-value={ks_pval:.4f}",
                         ha='center', fontsize=12)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save figure if output_dir is provided
            if output_dir is not None:
                # Create modality directory if needed
                mod_dir = os.path.join(output_dir, 
                                     modality_name if modality_name else "anndata")
                os.makedirs(mod_dir, exist_ok=True)
                
                # Save figure
                safe_coef = coef.replace(" ", "_").replace("/", "_").replace("\\", "_")
                file_path = os.path.join(mod_dir, f"pvalue_hist_{safe_coef}.png")
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                print(f"Saved histogram to {file_path}")
            
            # Store figure
            mod_figures[coef] = fig
        
        return mod_figures
    
    # Check if input is MuData or AnnData
    if hasattr(data, 'mod'):  # MuData object
        for modality in data.mod:
            print(f"Processing {modality}...")
            mod_figures = process_anndata(data[modality], modality_name=modality)
            if mod_figures:
                figures[modality] = mod_figures
    else:  # AnnData object
        figures['anndata'] = process_anndata(data)
    
    # Display figures if not saving
    if output_dir is None:
        for modality in figures:
            for coef in figures[modality]:
                plt.figure(figures[modality][coef].number)
                plt.show()
    
    print("Histogram generation complete.")
    return figures