import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from scipy.spatial.distance import norm, pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import kstest, boxcox
from sklearn.preprocessing import PowerTransformer


def plot_clustered_missing_pattern(phenotypes_df, max_missing=100, min_unique=10):
    """
    Visualize the missing data pattern for numeric phenotypes with <max_missing missing values
    and >min_unique unique values. Clusters both features and samples using Manhattan distance.
    """
    
    filtered_df = _select_continuous_measures(phenotypes_df, max_missing, min_unique)

    # Transpose: rows = variables, columns = samples
    missing_pattern = filtered_df.isnull().T
    # Cluster variables (rows)
    var_dist = pdist(missing_pattern, metric='cityblock')
    var_link = linkage(var_dist, method='average')
    var_order = leaves_list(var_link)
    ordered_vars = missing_pattern.index[var_order]
    # Cluster samples (columns)
    sample_dist = pdist(missing_pattern.T, metric='cityblock')
    sample_link = linkage(sample_dist, method='average')
    sample_order = leaves_list(sample_link)
    ordered_samples = missing_pattern.columns[sample_order]
    # Plot
    plt.figure(figsize=(min(20, 0.5*missing_pattern.shape[0]), 8))
    ax = sns.heatmap(
        missing_pattern.loc[ordered_vars, ordered_samples],
        cbar=False,
        xticklabels=False,
        cmap="viridis",
        linewidths=0
    )
    # Remove grid lines
    ax.grid(False)
    plt.title("Pattern of Missing Data (Filtered & Clustered Variables and Samples)")
    plt.ylabel("Variable")
    plt.xlabel("Sample")
    plt.tight_layout()
    plt.show()


def best_normalizing_transform(series):
    s = series.dropna()
    results = {}

    # Original
    stat, p = kstest((s - s.mean()) / s.std(), 'norm')
    results['original'] = {'stat': stat, 'p': p}

    # log2 (only for positive values)
    if (s > 0).all():
        s_log2 = np.log2(s)
        stat, p = kstest((s_log2 - s_log2.mean()) / s_log2.std(), 'norm')
        results['log2'] = {'stat': stat, 'p': p}
        # Box-Cox
        s_boxcox, _ = boxcox(s)
        stat, p = kstest((s_boxcox - s_boxcox.mean()) / s_boxcox.std(), 'norm')
        results['boxcox'] = {'stat': stat, 'p': p}
    else:
        results['log2'] = {'stat': np.nan, 'p': np.nan}
        results['boxcox'] = {'stat': np.nan, 'p': np.nan}

    # sqrt (only for non-negative values)
    if (s >= 0).all():
        s_sqrt = np.sqrt(s)
        stat, p = kstest((s_sqrt - s_sqrt.mean()) / s_sqrt.std(), 'norm')
        results['sqrt'] = {'stat': stat, 'p': p}
    else:
        results['sqrt'] = {'stat': np.nan, 'p': np.nan}

    # Yeo-Johnson (can handle negatives)
    try:
        pt = PowerTransformer(method='yeo-johnson')
        s_yeojohnson = pt.fit_transform(s.values.reshape(-1, 1)).flatten()
        stat, p = kstest((s_yeojohnson - s_yeojohnson.mean()) / s_yeojohnson.std(), 'norm')
        results['yeo-johnson'] = {'stat': stat, 'p': p}
    except Exception:
        results['yeo-johnson'] = {'stat': np.nan, 'p': np.nan}

    # Arcsinh (can handle negatives)
    s_arcsinh = np.arcsinh(s)
    stat, p = kstest((s_arcsinh - s_arcsinh.mean()) / s_arcsinh.std(), 'norm')
    results['arcsinh'] = {'stat': stat, 'p': p}

    # Find the best (highest p-value)
    best = max(results, key=lambda k: results[k]['p'] if not np.isnan(results[k]['p']) else -1)
    results['best'] = best

    return results


transform_func_map = {
    "original": lambda x: x,
    "log2": np.log2,
    "sqrt": np.sqrt,
    "boxcox": lambda x: pd.Series(boxcox(x.dropna())[0], index=x.dropna().index).reindex(x.index),
    "yeo-johnson": lambda x: pd.Series(PowerTransformer(method='yeo-johnson').fit_transform(x.dropna().values.reshape(-1, 1)).flatten(), index=x.dropna().index).reindex(x.index),
    "arcsinh": np.arcsinh
}


def transform_columns(df, transform_dict):
    """
    Transform columns in a DataFrame according to a dict of transformations.

    Args:
        df (pd.DataFrame): The input DataFrame.
        transform_dict (dict): Keys are column names, values are transformation functions (e.g., np.log2, np.sqrt).

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns transformed.
    """
    df_transformed = df.copy()
    for col, func in transform_dict.items():
        if col in df_transformed.columns:
            df_transformed[col] = func(df_transformed[col])
    return df_transformed


def plot_clustered_correlation_heatmap(df, figsize=(10, 8), cmap="coolwarm", annot=False, fmt=".2f"):
    """
    Compute and plot a hierarchically clustered correlation heatmap for a DataFrame.
    Clusters both rows and columns using correlation distance.
    """
    # Compute correlation matrix
    corr_matrix = df.corr()

    # Compute linkage for rows and columns
    # Use 1 - correlation as the distance metric
    pairwise_dists = 1 - corr_matrix
    # Replace NaN with 0 (perfect correlation) for clustering
    pairwise_dists = pairwise_dists.fillna(int(0))
    row_linkage = linkage(squareform(pairwise_dists), method='average')
    col_linkage = linkage(squareform(pairwise_dists.T), method='average')

    # Get the order of rows and columns
    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)
    ordered_corr = corr_matrix.iloc[row_order, col_order]

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        ordered_corr,
        cmap=cmap,
        vmin=-1, vmax=1,
        square=True,
        cbar_kws={"shrink": .8},
        annot=annot,
        fmt=fmt,
        linewidths=0
    )
    # Remove spines and grid lines
    for _, spine in ax.spines.items():
        spine.set_visible(False)
    ax.grid(False)
    plt.title("Hierarchically Clustered Pairwise Correlations")
    plt.tight_layout()
    plt.show()


def lower_diag_pairgrid_with_imputation(imputed_df, original_df, measures, palette=None, figsize=(12, 12), alpha=0.7):
    """
    Plot a lower-diagonal grid of scatterplots for all pairs in measures,
    coloring by pairwise imputation status (Observed, Imputed in one, Imputed in both).
    Diagonal shows histograms. Upper triangle is empty.
    """
    if palette is None:
        palette = {"Observed": "tab:blue", "Imputed in one": "tab:orange", "Imputed in both": "tab:red"}
    n = len(measures)
    fig, axes = plt.subplots(n, n, figsize=figsize)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i > j:
                x = imputed_df[measures[j]]
                y = imputed_df[measures[i]]
                # Pairwise imputation status
                imputed_x = original_df[measures[j]].isnull()
                imputed_y = original_df[measures[i]].isnull()
                status = (imputed_x.astype(int) + imputed_y.astype(int)).map({0: "Observed", 1: "Imputed in one"}).fillna("Imputed in both")
                plot_df = pd.DataFrame({
                    "x": x,
                    "y": y,
                    "Imputation Status": status
                })
                for label, color in palette.items():
                    mask = plot_df["Imputation Status"] == label
                    ax.scatter(plot_df.loc[mask, "x"], plot_df.loc[mask, "y"], label=label if (i == n-1 and j == 0) else "", color=color, alpha=alpha, s=20)
                if j == 0:
                    ax.set_ylabel(measures[i])
                else:
                    ax.set_yticklabels([])
                if i == n-1:
                    ax.set_xlabel(measures[j])
                else:
                    ax.set_xticklabels([])
            elif i == j:
                # Diagonal: histogram
                ax.hist(imputed_df[measures[i]], bins=20, color="gray", alpha=0.7)
                if j == 0:
                    ax.set_ylabel(measures[i])
                if i == n-1:
                    ax.set_xlabel(measures[j])
            else:
                ax.axis("off")
    # Only add legend to the top-right plot (axes[0, -1])
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=8) for label, color in palette.items()]
    axes[0, -1].legend(handles=handles, title="Imputation Status", loc='upper right', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show() 


def _select_continuous_measures(phenotypes_df, max_missing=100, min_unique=10):
    """
    Select numeric columns from a DataFrame with <max_missing missing values
    and >min_unique unique values.
    """
    # Select numeric columns
    numeric_df = phenotypes_df.select_dtypes(include=[np.number])
    # Filter columns
    filtered_cols = [
        col for col in numeric_df.columns
        if numeric_df[col].isnull().sum() < max_missing and numeric_df[col].nunique(dropna=True) > min_unique
    ]
    filtered_df = numeric_df[filtered_cols]
    if filtered_df.empty:
        print("No variables meet the filtering criteria.")
        return
    
    return filtered_df