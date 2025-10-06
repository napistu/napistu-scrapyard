""" Utilities for ETLing the AIDOCell foundation model. """

import json
import logging
import numpy as np
import os
import pandas as pd
import torch
from types import SimpleNamespace

import modelgenerator.cell.utils as cell_utils

# local .py file
from utils import (
    ONTOLOGIES,
    RESULTS_DEFS,
)

logger = logging.getLogger(__name__)

AIDOCELL_DEFS = SimpleNamespace(
    MODEL_NAME = "AIDOCell",
    # files
    GENE_FILE = "gene_lists/OS_scRNA_gene_index.19264.tsv",
    # parameters
    EMBED_DIM = "embed_dim",
    N_LAYERS = "n_layers",
    N_HEADS = "n_heads",
    HIDDEN_DIM = "hidden_dim",
)

def load_aidocell_model(model_class):
    """
    Load AIDOCell model in eval mode
    
    Parameters
    ----------
    model_class : class
        AIDOCell model class to load
        
    Returns
    -------
    model
        The AIDOCell model in eval mode
    """
    
    model = model_class(
        legacy_adapter_type=None,
        default_config=None,
        from_scratch=False
    )
    model.eval()
    return model


def load_gene_annotations():
    """
    Load gene annotations from AIDOCell model

    This is a flat file which is bundled with the package
    
    Returns
    -------
    pd.DataFrame
        DataFrame with gene annotations
    """
    
    load_base = os.path.dirname(os.path.abspath(cell_utils.__file__))
    gene_file = os.path.join(load_base, AIDOCELL_DEFS.GENE_FILE)
    
    # Load gene symbols
    gene_symbols = pd.read_csv(gene_file, sep='\t')['gene_name'].values
    
    # Build the mapping from symbols to Ensembl IDs
    gene_map = cell_utils.build_map(gene_symbols)
    
    # Create the mapping table
    gene_table = pd.DataFrame({
        RESULTS_DEFS.VOCAB_NAME: gene_symbols,
        ONTOLOGIES.SYMBOL: gene_symbols,
        ONTOLOGIES.ENSEMBL_GENE: [gene_map.get(x, f'{x}_unknown_ensg') for x in gene_symbols]
    })
    
    return gene_table


def _extract_attention_weights(model):
    """
    Extract core attention weights (Q, K, V, O) from all layers
    
    Parameters
    ----------
    model : AIDOCell model
        The AIDOCell model
        
    Returns
    -------
    dict
        Dictionary with attention weights by layer
    """
    attention_weights = {}
    encoder = model.encoder
    transformer_layers = encoder.encoder.layer
    n_layers = model.get_num_layer()
    
    for layer_idx in range(n_layers):
        layer = transformer_layers[layer_idx]
        attention_self = layer.attention.self
        attention_output = layer.attention.output
        
        attention_weights[RESULTS_DEFS.LAYER_NAME_TEMPLATE.format(layer_idx=layer_idx)] = {
            RESULTS_DEFS.W_Q: attention_self.query.weight.detach().cpu().numpy(),
            RESULTS_DEFS.W_K: attention_self.key.weight.detach().cpu().numpy(),
            RESULTS_DEFS.W_V: attention_self.value.weight.detach().cpu().numpy(),
            RESULTS_DEFS.W_O: attention_output.dense.weight.detach().cpu().numpy(),
        }
    
    return attention_weights


def extract_model_weights(model):
    """
    Extract model weights in the standardized format
    
    Parameters
    ----------
    model : AIDOCell model
        The AIDOCell model
        
    Returns
    -------
    dict
        Dictionary containing gene_embedding and attention_weights
    """
    
    # Extract gene embeddings
    encoder = model.encoder
    n_genes = len(load_gene_annotations())
    
    with torch.no_grad():
        gene_positions = torch.arange(n_genes)
        gene_embedding = encoder.position_embedding(gene_positions).cpu().numpy()
    
    # Extract attention weights
    attention_weights = _extract_attention_weights(model)
    
    return {
        RESULTS_DEFS.GENE_EMBEDDING: gene_embedding,
        RESULTS_DEFS.ATTENTION_WEIGHTS: attention_weights
    }


def _format_model_metadata(model):
    """
    Extract model architecture metadata
    
    Parameters
    ----------
    model : AIDOCell model
        The AIDOCell model
        
    Returns
    -------
    dict
        Dictionary with model metadata
    """
    encoder = model.encoder
    gene_annotations = load_gene_annotations()
    n_genes = len(gene_annotations)
    
    # Get vocabulary as list of gene symbols (AIDOCell doesn't have special tokens)
    vocab_list = gene_annotations[RESULTS_DEFS.VOCAB_NAME].tolist()
    
    return {
        RESULTS_DEFS.MODEL_NAME: AIDOCELL_DEFS.MODEL_NAME,
        RESULTS_DEFS.N_GENES: n_genes,
        RESULTS_DEFS.N_VOCAB: n_genes,  # Same as n_genes for AIDOCell (no special tokens)
        RESULTS_DEFS.ORDERED_VOCABULARY: vocab_list,
        RESULTS_DEFS.EMBED_DIM: int(model.get_embedding_size()),
        RESULTS_DEFS.N_LAYERS: int(model.get_num_layer()),
        RESULTS_DEFS.N_HEADS: int(encoder.config.num_attention_heads),
        # Additional AIDOCell-specific metadata
        AIDOCELL_DEFS.HIDDEN_DIM: int(encoder.config.hidden_size),
    }


def load_aidocell(model_class):
    """
    Load the AIDOCell model and return model, gene annotations, and metadata
    
    Parameters
    ----------
    model_class : class
        AIDOCell model class to load
        
    Returns
    -------
    model : AIDOCell model
        The AIDOCell model
    gene_annotations : pandas.DataFrame
        DataFrame with gene annotations
    model_metadata : dict
        Dictionary with model metadata
    """
    
    logger.info("Loading AIDOCell model")
    model = load_aidocell_model(model_class)
    
    logger.info("Loading gene annotations")
    gene_annotations = load_gene_annotations()
    
    logger.info("Formatting model metadata")
    model_metadata = _format_model_metadata(model)
    
    return model, gene_annotations, model_metadata
