""" Utilities for ETLing the scPRINT foundation model. """

import logging
import os
from types import SimpleNamespace

import bionty as bt
import torch
import numpy as np
import pandas as pd
from scdataloader.utils import populate_my_ontology
from scprint import scPrint

# local .py file
from etl_utils import (
    MODELS,
    ONTOLOGIES,
    RESULTS_DEFS,
)

logger = logging.getLogger(__name__)

SCPRINT_DEFS = SimpleNamespace(
    MODEL_NAME = MODELS.SCPRINT,
    # files
    CHECKPOINT_FILENAME = "v2-medium.ckpt",
    # parameters
    D_MODEL = "d_model",
    N_LAYERS = "nlayers",
    N_HEADS = 4,  # Fixed architecture parameter
)

def load_scprint_model(checkpoint_path, transformer="normal"):

    """
    Load scPRINT model

    Parameters
    ----------
    checkpoint_path : str
        Path to the scPRINT checkpoint file
    transformer : str, optional
        Transformer type, by default "normal"

    Returns
    -------
    scPrint
        The scPRINT model
    """


    """Load scPRINT model in eval mode"""
    m = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    if "prenorm" in m['hyper_parameters']:
        m['hyper_parameters'].pop("prenorm")
        
    if "label_counts" in m['hyper_parameters']:
        model = scPrint.load_from_checkpoint(
            checkpoint_path, 
            precpt_gene_emb=None, 
            classes=m['hyper_parameters']['label_counts'], 
            transformer=transformer
        )
    else:
        model = scPrint.load_from_checkpoint(
            checkpoint_path, 
            precpt_gene_emb=None, 
            transformer=transformer
        )
    
    model.eval()
    return model


def load_gene_annotations(model) -> pd.DataFrame:

    """
    Load gene annotations from scPRINT model

    Parameters
    ----------
    model : scPrint
        The scPRINT model

    Returns
    -------
    pd.DataFrame
        DataFrame with gene annotations
    """
    
    gene_table = pd.DataFrame({
        RESULTS_DEFS.VOCAB_NAME: model.genes,
        ONTOLOGIES.ENSEMBL_GENE: model.genes,
    })
    
    # Optionally add gene symbols from lamindb
    try:
        all_genes_df = bt.Gene.filter().df()
        ensembl_to_symbol = all_genes_df.set_index('ensembl_gene_id')['symbol'].to_dict()
        gene_table[ONTOLOGIES.SYMBOL] = gene_table[ONTOLOGIES.ENSEMBL_GENE].map(ensembl_to_symbol)
    except:
        gene_table[ONTOLOGIES.SYMBOL] = gene_table[ONTOLOGIES.ENSEMBL_GENE]
    
    return gene_table


def populate_lamin_db() -> None:

    """
    Populate the lamin database

    Add species, identifiers, and other metadata to the lamin database

    Returns
    -------
    None
    """
    
    # quick check to see if the lamin database is already configured
    organisms = bt.Organism.filter().df()
    human_defined = 'NCBITaxon:9606' in organisms['ontology_id'].values if len(organisms) > 0 else False
    if not human_defined:
        logger.info("Populating the full metadata catalog recommended by the scPRINT developers")
        # populate the full metadata catalog recommended by the scPRINT developers
        populate_my_ontology() 
    else:
        logger.info("Lamin database already configured")


def extract_model_weights(model):
    """
    Extract model weights in the standardized format
    
    Parameters
    ----------
    model : scPrint
        The scPRINT model
        
    Returns
    -------
    dict
        Dictionary containing gene_embedding and attention_weights
    """
    
    # Extract gene embeddings
    gene_embedding = model.gene_encoder.embeddings.weight.detach().cpu().numpy()
    
    # Extract attention weights
    attention_weights = _extract_attention_weights(model)
    
    return {
        RESULTS_DEFS.GENE_EMBEDDING: gene_embedding,
        RESULTS_DEFS.ATTENTION_WEIGHTS: attention_weights
    }


def load_scprint(checkpoint_path, transformer="normal"):
    """
    Load the scPRINT model and return model, gene annotations, and metadata
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the scPRINT checkpoint file
    transformer : str, optional
        Transformer type, by default "normal"
        
    Returns
    -------
    model : scPrint
        The scPRINT model
    gene_annotations : pandas.DataFrame
        DataFrame with gene annotations
    model_metadata : dict
        Dictionary with model metadata
    """
    
    logger.info("Loading scPRINT model")
    model = load_scprint_model(checkpoint_path, transformer)
    
    logger.info("Loading gene annotations")
    gene_annotations = load_gene_annotations(model)
    
    logger.info("Formatting model metadata")
    model_metadata = _format_model_metadata(model)
    
    return model, gene_annotations, model_metadata

# private utils

def _extract_attention_weights(model):
    """Extract self-attention weights (Q, K, V, O) from all layers"""
    attention_weights = {}
    d_model = model.d_model
    n_layers = model.nlayers
    
    for layer_idx in range(n_layers):
        block = model.transformer.blocks[layer_idx]
        mixer = block.mixer
        
        # Get combined QKV weight: (3 * d_model, d_model)
        qkv_weight = mixer.Wqkv.weight.detach().cpu().numpy()
        
        attention_weights[RESULTS_DEFS.LAYER_NAME_TEMPLATE.format(layer_idx=layer_idx)] = {
            RESULTS_DEFS.W_Q: qkv_weight[:d_model, :],
            RESULTS_DEFS.W_K: qkv_weight[d_model:2*d_model, :],
            RESULTS_DEFS.W_V: qkv_weight[2*d_model:, :],
            RESULTS_DEFS.W_O: mixer.out_proj.weight.detach().cpu().numpy(),
        }
    
    return attention_weights


def _format_model_metadata(model):
    """Extract model architecture metadata"""
    
    # Get vocabulary as list of genes (scPRINT doesn't have special tokens)
    vocab_list = list(model.genes)
    n_genes = len(vocab_list)
    
    return {
        RESULTS_DEFS.MODEL_NAME: SCPRINT_DEFS.MODEL_NAME,
        RESULTS_DEFS.N_GENES: n_genes,
        RESULTS_DEFS.N_VOCAB: n_genes,  # Same as n_genes for scPRINT (no special tokens)
        RESULTS_DEFS.ORDERED_VOCABULARY: vocab_list,
        RESULTS_DEFS.EMBED_DIM: int(model.d_model),
        RESULTS_DEFS.N_LAYERS: int(model.nlayers),
        RESULTS_DEFS.N_HEADS: SCPRINT_DEFS.N_HEADS
    }