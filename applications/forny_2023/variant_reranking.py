# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: forny-2023
#     language: python
#     name: forny-2023
# ---

# %% [markdown]
# # Forny 2023 — Multi-Omics Variant Re-ranking
#
# ## Background
#
# This notebook re-ranks gene-level variant prioritizations from [Exomiser](https://exomiser.readthedocs.io)
# by integrating multi-omics network evidence from **Napistu**.
#
# The cohort (Forny 2023) consists of patients with methylmalonic acidemia (MMA) and related organic
# acidemias, split into three clinical subgroups:
# - **mut** (MMA001–MMA150): confirmed disease-causing variants
# - **undiagnosed** (MMA151–MMA210): molecularly unresolved patients
# - **unaffected** (MMA211–MMA230): healthy controls
#
# ## Analysis strategy
#
# 1. Load Exomiser gene-level variant rankings (phenotype + pathogenicity combined score)
# 2. Load Napistu multi-omics driver scores (proteomics + transcriptomics network enrichment)
# 3. Merge and compute a combined rank by averaging Exomiser rank and multi-omics attribute rank
# 4. Evaluate whether re-ranking improves retrieval of known causal genes
# 5. Inspect top candidates in undiagnosed patients

# %% [markdown]
# ## Setup
# Dependencies extend the `predict_driver_mutations` environment with `plotnine` for ggplot2-style plots.

# %%
# Environment: same as predict_driver_mutations with the addition:
# - plotnine: for plotting
# !uv pip install plotnine

# %%
import logging
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import plotnine as pn


logging.basicConfig(level=logging.INFO)


# %% [markdown]
# ## Configuration
#
# Paths and parameters are centralised in `inputs` and `params` to make the notebook
# easy to adapt without touching analysis logic.

# %%
class inputs:  # type: ignore
    _fol_input = Path("../../data/Forny2023/input")
    # not published gene level exomiser prioritization
    fn_exomiser_genes = _fol_input / "exomiser_genes_v1.csv"
    # known variants based on the published variants in the supplementary
    fn_known = _fol_input / '42255_2022_720_MOESM1_ESM_sup_tab1.csv'
    # generated from predict_driver_mutations
    fn_napistu = _fol_input / "../interim/mma_driver_mutation_rankings.parquet"


class params:  # type: ignore
    mma_group_dict = {
        'mut': (1, 150),
        'undiagnosed': (151, 210),
        'unaffected': (211, 230),
    }



# %%
class V:
    # Common / Shared
    sample_id = 'sample_id'              # Unique identifier for each sample
    sample_subgroup = 'subgroup'         # Clinical subgroup (e.g., mut, undiagnosed)
    symbol = 'symbol'                    # Gene symbol
    
    # Known variants (dat_known)
    variant_hgnc_symbol = 'variant_hgnc_symbol' # Known disease-causing gene symbol
    variant_1_nt = 'variant_1_nt'        # Nucleotide variant information
    
    # Multi-omics drivers (dat_mma_driver)
    patient = 'patient'                  # Patient ID (matches sample_id)
    modality = 'modality'                # Multi-omics modality (e.g., genomics, proteomics)
    attribute_rank = 'attribute_rank'    # Rank of the gene in the multi-omics network
    log2_enrichment = 'log2_enrichment'  # Log2 enrichment score from multi-omics
    
    # Exomiser results (dat_exom)
    exomiser_gene_symbol = 'GENE_SYMBOL'          # Exomiser gene symbol
    exomiser_rank = 'RANK'                        # Exomiser variant rank
    exomiser_pipeline = 'pipeline'                # Variant calling pipeline used
    exomiser_id = 'ID'                            # Exomiser variant ID
    exomiser_p_value = 'P-VALUE'                  # Exomiser p-value
    exomiser_score = 'EXOMISER_GENE_COMBINED_SCORE' # Combined Exomiser score
    
    # Engineered / Computed features
    known_gene = 'known_gene'            # Mapped known disease gene for a sample
    known_symbol = 'known_symbol'        # Renamed known gene symbol for merging
    is_known = 'is_known'                # Boolean flag if gene matches known gene
    combined_rank = 'combined_rank'      # Average of exomiser rank and multi-omics rank
    re_rank = 're_rank'                  # Rank of the combined_rank within each sample/pipeline
    delta_rank = 'delta_rank'            # Difference between re_rank and original exomiser RANK
    attribute_rank_min = 'attribute_rank.min' # Minimum attribute rank across modalities
    log2_enrichment_max = 'log2_enrichment.max' # Maximum log2 enrichment across modalities
    log2_enrichment_prot = 'log2_enrichment.proteomics'
    log2_enrichment_trans = 'log2_enrichment.transcriptomics'
    is_retrieved = 'is_retrieved'        # Boolean if gene has multi-omics data


# %% [markdown]
# ## Helper functions
#
# `range_dict_to_id_mapping` converts the compact integer-range group definitions in `params`
# to a flat `{sample_id: group}` lookup dict, which is used to annotate every dataframe with
# the clinical subgroup.
#
# `add_known` / `add_subgroup` are thin wrappers that keep the main pipeline readable.

# %%
# Helper functions
def range_dict_to_id_mapping(
    group_ranges: dict[str, tuple[int, int]],
    id_format="MMA{:03d}",
) -> dict[str, str]:
    """Converts a range dict to an id mapping

    Args:
        group_ranges (dict[str, tuple[int, int]]): A mapping from groups to ranges
            in the form {<groupname>: (startidx, stopidx)}
            with start_idx and stop_idx being the id range including start and stop
        id_format (str, optional): Format of the identifier. Defaults to "MMA%03d".

    Returns:
        dict[str, int]: dict mapping an id to a group

    Example:
        >>> range_dict_to_id_mapping({'group_a': (1, 2), 'group_b': (3, 3)})
        {'MMA001': 'group_a', 'MMA002': 'group_a', 'MMA003': 'group_b'}
    """
    return {
        **{
            id_format.format(i): g
            for g, sample_range in group_ranges.items()
            for i in np.arange(sample_range[0], sample_range[1] + 1)
        },
    }

def add_known(dat, kdict):
    return dat.assign(known_gene= lambda x:  x[V.sample_id].map(kdict))


def add_subgroup(x, group_dict: dict[str, str]):
    x=  x.assign(**{V.sample_subgroup: lambda x: x[V.sample_id].map(group_dict)})
    x[V.sample_subgroup] = x[V.sample_subgroup].astype(pd.CategoricalDtype(list(params.mma_group_dict.keys())))
    return x


# %% [markdown]
# ## Data loading
#
# Three datasets are loaded:
# | Dataset | Source | Purpose |
# |---|---|---|
# | `dat_known` | Published supplementary table | Ground-truth causal variants |
# | `dat_exom` | Unpublished Exomiser output | Variant prioritization per patient |
# | `dat_mma_driver` | Napistu output | Multi-omics network driver scores |


# %%
dat_known = pd.read_csv(inputs.fn_known)
dat_known.head()


# %%
known_dict = dat_known.dropna(subset=V.variant_hgnc_symbol).set_index(V.sample_id)[V.variant_hgnc_symbol].to_dict()
samples_w_variants = dat_known.query(f'{V.variant_1_nt}.notna()')[V.sample_id]

# %%

group_dict = range_dict_to_id_mapping(params.mma_group_dict)
dat_exom = (pd.read_csv(inputs.fn_exomiser_genes)
            .pipe(add_subgroup, group_dict=group_dict)
)


# %% [markdown]
# Quick sanity check: how many patients per clinical subgroup are present in the Exomiser results?

# %%
(dat_exom
    .groupby([V.sample_subgroup])[V.sample_id].nunique()
)

# %%
dat_exom[V.sample_subgroup]

# %%
dat_mma_driver = pd.read_parquet(inputs.fn_napistu)

# %%
# Look at example
dat_mma_driver.query(f'{V.symbol} == "ACSF3"').sort_values(V.attribute_rank)


# %%



# %%
known_genes = dat_known[V.variant_hgnc_symbol]

dat_exom = dat_exom.pipe(add_known, known_dict)

# %%
dat_mma_driver_filtered_known = dat_mma_driver.query(f'{V.symbol} in @known_genes').sort_values([V.patient, V.attribute_rank])

# %%
dat_mma_driver_filtered_known.query(f'{V.patient} == "MMA205"')

# %%
dat_mma_driver[V.attribute_rank].max()

# %%
(dat_mma_driver_filtered_known
    .assign(mma_nr   = lambda x: x[V.patient].str[3:].astype(int))
.query(f'mma_nr > 149 & {V.attribute_rank} < 300')
)

# %%
# Info log
logging.info(f'Number of MMA patients with variants in dat_exom: %d', dat_exom[V.sample_id].nunique())
logging.info(f'Number of MMA patients with variants in dat_mma_driver: %d', dat_mma_driver[V.patient].nunique())

# %% [markdown]
# ## Multi-omics context for known causal genes
#
# Here we look at how the known causal genes rank in the Napistu multi-omics network
# across all patients — regardless of Exomiser evidence. This helps assess how broadly
# the network "sees" known MMA-related genes and whether the signal is modality-specific.
# note: as the multi-omics network data is only for the main network module, it is sparse and does not contain all genes!
# %%
known_genes = dat_known[V.variant_hgnc_symbol].unique()

# %%
dat_mma_driver

# %%
pdat =(dat_mma_driver.merge(dat_known
                     .rename(columns={V.variant_hgnc_symbol: V.known_symbol}),
                     left_on=[V.patient],
                     right_on=[V.sample_id], how='outer')
    .dropna(subset=V.known_symbol)
    .query(f'{V.symbol} in @known_genes')
     .eval(f'{V.is_known} = {V.symbol} == {V.known_symbol}')
      )


# %%
(pdat >>
 pn.ggplot(pn.aes(x=V.symbol, y=V.attribute_rank, color=V.is_known))
      + pn.facet_grid(f'{V.modality}~.')
     + pn.geom_point(position=pn.position_jitterdodge())
     +pn.scale_y_log10()
     +pn.coord_trans(y="reverse")
)

# %%
(pdat >>
 pn.ggplot(pn.aes(x=V.symbol, y=V.log2_enrichment, color=V.is_known))
      + pn.facet_grid(f'{V.modality}~.')
     + pn.geom_point(position=pn.position_jitterdodge())
     +pn.scale_y_log10()
)

# %% [markdown]
# ### Per-patient, per-modality enrichment matrix
#
# Wide-format view of `log2_enrichment` for all known causal genes across patients and modalities.
# used to get an idea of the sparsity

# %%
dat_mma_driver.pivot_table(index=[V.patient, V.modality],
                           columns=V.symbol,
                           values=V.log2_enrichment)

# %% [markdown]
# ## Multi-omics driver summarization
#
# Napistu outputs one row per *(patient, gene, modality)* triplet.
# `summarize_drivers` collapses this to *(patient, gene)* by:
# - taking the **maximum** `log2_enrichment` across modalities (best network evidence)
# - taking the **minimum** `attribute_rank` across modalities (best network rank)
#
# Individual modality columns (proteomics, transcriptomics) are retained for later inspection.

# %%

def summarize_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize driver mutation rankings by pivoting across modalities.
    
    Computes the maximum log2 enrichment and minimum attribute rank
    across all available modalities for each patient and gene symbol.
    
    Args:
        df: DataFrame containing 'patient', 'symbol', 'modality',
            'log2_enrichment', and 'attribute_rank'.
            
    Returns:
        pd.DataFrame: Pivoted summary with flattened column names.
    """
    df_sum = df.pivot_table(
        index=[V.patient, V.symbol],
        columns=[V.modality],
        values=[V.log2_enrichment, V.attribute_rank]
    )

    df_sum[(V.log2_enrichment, 'max')] = df_sum[V.log2_enrichment].max(axis=1)
    df_sum[(V.attribute_rank, 'min')] = df_sum[V.attribute_rank].min(axis=1)
    
    df_sum.columns = [f"{c[0]}.{c[1]}" for c in df_sum.columns]
    
    return df_sum.reset_index()

dat_mma_driver_sum = summarize_drivers(dat_mma_driver)
dat_mma_driver_sum


# %% [markdown]
# ## Merging Exomiser and multi-omics evidence
#
# Exomiser was run with 3 separate variant calling pipelines:
# - **exomiser**: standard SNV/indel pipeline based on deepvariant calls
# - **exomiser_svc**: addition of structural variants to the standard pipeline
# - **genomiser**: includes intronic/upstream/downstream and CNV variant calls
#
# We keep all pipelines and stratify all downstream comparisons by pipeline.
# Genes without Napistu coverage receive a high penalty rank (`fill_rank=6000`).

# %%
dat_exom[V.exomiser_pipeline].unique()

# %%
dat_genom = dat_exom.query(f'{V.exomiser_pipeline} == "exomiser"')

# %%
(dat_exom
    .sort_values([V.exomiser_rank],
                ascending=True)
    .query(f'{V.exomiser_pipeline} == "exomiser"')
     .query(f'{V.sample_id} == "MMA023"')
)

# %%
def merge_variants(
    df_exom: pd.DataFrame, 
    df_driver_sum: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges exomiser variant rankings with multi-omics driver summaries.
    
    Args:
        df_exom: Exomiser variant dataframe.
        df_driver_sum: Multi-omics driver summary dataframe.
        
    Returns:
        pd.DataFrame: Merged dataframe containing both exomiser and multi-omics data.
    """
    return (
        df_exom
        # remove duplicates on the gene level
        # usually this is because AR and AD are listed twice
        .sort_values([V.exomiser_rank], ascending=True)
        .drop_duplicates(subset=[V.sample_id, V.exomiser_pipeline, V.exomiser_gene_symbol])
        # Outer merge to keep all exomiser variants
        .merge(
            df_driver_sum, 
            left_on=[V.sample_id, V.exomiser_gene_symbol],
            right_on=[V.patient, V.symbol],
            how='left'
        )
        # make sure symbol is the not NaN
        .assign(**{V.symbol: lambda x: x[V.exomiser_gene_symbol]})
    )

def rerank_variants(
    df_merged: pd.DataFrame, 
    fill_rank: int = 6000
) -> pd.DataFrame:
    """
    Calculates an updated combined rank based on exomiser and multi-omics ranks.
    
    Args:
        df_merged: Merged dataframe containing 'RANK' and 'attribute_rank.min'.
        fill_rank: Rank to fill for genes without multi-omics data.
        
    Returns:
        pd.DataFrame: Dataframe with new rank metrics ('combined_rank',
            're_rank', and 'delta_rank').
    """
    df_reranked = df_merged.copy()
    
    df_reranked[V.combined_rank] = df_reranked[[V.exomiser_rank, V.attribute_rank_min]].fillna(fill_rank).mean(axis=1)
    
    df_reranked[V.re_rank] = (
        df_reranked.groupby([V.sample_id, V.exomiser_pipeline])[V.combined_rank]
        .rank(method='min')
    )
    
    df_reranked[V.delta_rank] = df_reranked[V.re_rank] - df_reranked[V.exomiser_rank]
    
    return df_reranked

dat_merged = merge_variants(dat_exom, dat_mma_driver_sum)
dat_merged = rerank_variants(dat_merged)

# %% [markdown]
# ## Benchmarking: known causal genes
#
# `dat_hits` restricts to:
# 1. Rows where the Exomiser gene matches the known causal gene — measures retrieval quality.
# 2. Rows where no known gene exists (undiagnosed) — potential novel discoveries.
#
# **Key metric:** `delta_rank = re_rank − exomiser_rank`  
# Negative values mean the combined ranking **improves** on Exomiser alone.

# %%
dat_merged[[V.sample_id, V.exomiser_rank, V.combined_rank, V.attribute_rank_min]]

# %%
dat_merged.query(f'{V.exomiser_pipeline} == "exomiser"')[[V.sample_id, V.exomiser_id]]

# %%
dat_merged

# %%
dat_hits = dat_merged.query(f'{V.exomiser_gene_symbol} == {V.known_gene} | {V.known_gene}.isna()')

    

# %%
dat_hits[V.symbol]

# %%
# get the change in rank for known

(dat_hits.query(f'{V.symbol} == {V.known_gene}')
     .eval(f'{V.delta_rank} = {V.re_rank}-{V.exomiser_rank}')
     >> pn.ggplot(pn.aes(x=V.delta_rank))
     + pn.facet_grid(f'{V.exomiser_pipeline}~.')
     +pn.geom_histogram()
     + pn.theme(figure_size=(5,6),
               strip_text_y=pn.element_text(angle=0))
     + pn.ggtitle('Change in rank of\n'
                  'known disease causing '
                  'genes\n'
                  'after multi-omics\nnetwork integration')
     + pn.coord_trans(x='reverse')
 + pn.xlab('Rank change [lower = better]')
)


# %%
dat_hits[V.sample_subgroup]

# %%
dat_exom

# %%
(dat_exom.groupby([V.sample_subgroup, V.known_gene])[V.sample_id].nunique())

# %%
(dat_hits.query(f'{V.exomiser_gene_symbol} == {V.known_gene}')
    .rename(columns={V.exomiser_rank: 'old\nrank',
                    V.re_rank: 'new\nrank'    
                })
     .melt(id_vars=[V.sample_subgroup, 
                            V.sample_id, V.exomiser_gene_symbol,
                   V.exomiser_pipeline],
                  
                  value_vars=['old\nrank', 'new\nrank']).reset_index()
     
     >> pn.ggplot(pn.aes(x='variable', y='value' , color=V.exomiser_gene_symbol))
     + pn.facet_grid(f'{V.exomiser_pipeline}~{V.sample_subgroup}')
     +pn.geom_jitter(width=0.1, height=0)
     + pn.geom_line(pn.aes(group=f'{V.sample_id}+{V.exomiser_pipeline}'), alpha=0.2)
     + pn.theme(figure_size=(5,6),
               strip_text_y=pn.element_text(angle=0))
     + pn.scale_y_log10()
 + pn.coord_trans(y='reverse', x='reverse')
     + pn.ggtitle('Change in rank of\n'
                  'known disease causing '
                  'genes\n'
                  'after multi-omics\nnetwork integration')
 + pn.xlab('Improvement with multi-omics network')
 + pn.ylab('Rank of known disease causing gene')
)

# %%
dat_hits.groupby(V.known_gene)[V.sample_id].nunique()

# %%
(dat_merged
    .eval(f'{V.is_retrieved} = `{V.attribute_rank_min}`.notna()')
    .query(f'{V.exomiser_gene_symbol} == {V.known_gene}')
    
    .groupby([V.is_retrieved, V.known_gene])[V.sample_id].nunique())

# %%
(dat_merged
    .eval(f'{V.is_retrieved} = `{V.attribute_rank_min}`.notna()')
    .query(f'{V.exomiser_gene_symbol} == {V.known_gene}')
)

# %%
(dat_hits.query(f'{V.exomiser_gene_symbol} == {V.known_gene}')
    .query(f'{V.exomiser_pipeline} == "genomiser"')
     .sort_values(V.delta_rank)
     [[
         V.sample_id, V.exomiser_rank, V.exomiser_id, V.exomiser_gene_symbol, V.re_rank, V.delta_rank,
         V.attribute_rank_min, 
         V.log2_enrichment_max
     ]]
 
)

# %% [markdown]
# ## Undiagnosed patients — novel candidate discovery
#
# For patients without a published causal gene we inspect the top re-ranked candidates
# (combined `re_rank < 15`) to identify genes that rise due to corroborating multi-omics
# network evidence. These are the most actionable outputs of the pipeline.
#
# Selected patients are inspected individually below.

# %%
dat_new_hits = (dat_hits.query(f'{V.known_gene}.isna()')

 .query(f'{V.re_rank} < 15')
[[V.exomiser_pipeline, V.sample_id, V.symbol, V.log2_enrichment_prot,
  V.log2_enrichment_trans,
  V.log2_enrichment_max,
  V.attribute_rank_min, 	
  V.exomiser_rank,
  V.combined_rank,
  V.re_rank, V.exomiser_id, V.exomiser_p_value, V.exomiser_score
 ]].sort_values([V.sample_id, V.re_rank])

)
(dat_new_hits    .query(f'{V.exomiser_pipeline} == "genomiser"')
    
.pivot_table(index=[V.sample_id], values=V.exomiser_id, columns=V.re_rank, aggfunc=list)
)

# %% [markdown]
# MMA 153 -> SUCLG1

# %%
(dat_new_hits
    .query(f'{V.sample_id} == "MMA088"')
    .head(20)
)


# %%
(dat_new_hits
    .query(f'{V.sample_id} == "MMA139"')
     .query(f'{V.symbol} == "MMUT"')
)


# %%
(dat_new_hits
    .query(f'{V.sample_id} == "MMA153"')
    .head(10)
)

# %%
(dat_new_hits
    .query(f'{V.sample_id} == "MMA157"')
    
    .head(20)
)

# %%
(dat_new_hits
    .query(f'{V.sample_id} == "MMA159"')
    .head(20)
)

# %%
(dat_new_hits
    .query(f'{V.sample_id} == "MMA161"')
    .head(20)
)

# %%
(dat_new_hits
    .query(f'{V.sample_id} == "MMA162"')
    .head(20)
)

# %%
(dat_new_hits
    .query(f'{V.sample_id} == "MMA163"')
    .head(20)
)

# %%
(dat_new_hits
    .query(f'{V.sample_id} == "MMA169"')
    .head(20)
)

# %%
