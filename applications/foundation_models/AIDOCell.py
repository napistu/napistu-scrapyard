""" Utilities for ETLing the AIDOCell foundation model. """

import logging
import numpy as np
import os
import pandas as pd
import torch
from types import SimpleNamespace
from typing import List

import modelgenerator.cell.utils as cell_utils

# Import from napistu_torch
from napistu.constants import ONTOLOGIES
from napistu_torch.load.constants import (
    FM_DEFS,
    FOUNDATION_MODEL_NAMES,
)
from napistu_torch.load.foundation_models import (
    AttentionLayer,
    FoundationModel,
    FoundationModelWeights,
)

logger = logging.getLogger(__name__)

AIDOCELL_DEFS = SimpleNamespace(
    MODEL_NAME = FOUNDATION_MODEL_NAMES.AIDOCELL,
    # files
    GENE_FILE = "gene_lists/OS_scRNA_gene_index.19264.tsv",
    PREFIX_TEMPLATE = "{model_name}_{model_class_name}",
    # parameters
    EMBED_DIM = "embed_dim",
    N_LAYERS = "n_layers",
    N_HEADS = "n_heads",
    HIDDEN_DIM = "hidden_dim",
)


def process_model(model_class, output_dir) -> None:

    """
    Process a given model class and save the results to the output directory.

    Parameters
    ----------
    model_class : class
        AIDOCell model class to load
    output_dir : str
        Output directory to save the results
        
    Returns
    -------
    None
    """

    model_class_name = model_class.__name__
    file_prefix = AIDOCELL_DEFS.PREFIX_TEMPLATE.format(model_name=AIDOCELL_DEFS.MODEL_NAME, model_class_name=model_class_name)
    
    logger.info(f"Extracting: {model_class_name}")
    
    # 1. Load model and data
    logger.info("\n1. Loading model and data...")
    model, gene_annotations, model_metadata = load_aidocell(model_class)
    logger.info(f"   {len(gene_annotations)} genes, {model_metadata[FM_DEFS.N_LAYERS]} layers")

    # 2. Extract weights
    logger.info("2. Extracting weights...")
    weights = extract_model_weights(model)
    logger.info(f"   Embeddings: {weights.gene_embedding.shape}")
    logger.info(f"   Attention weights: {model_metadata[FM_DEFS.N_LAYERS]} layers × 4 matrices (Q,K,V,O)")

    # 3. Create FoundationModel and save
    logger.info("3. Creating FoundationModel and saving...")
    foundation_model = FoundationModel(
        weights=weights,
        gene_annotations=gene_annotations,
        model_metadata=model_metadata,
    )
    foundation_model.save(output_dir, file_prefix)
    logger.info("   Successfully saved all results!")

    return None


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
        FM_DEFS.VOCAB_NAME: gene_symbols,
        ONTOLOGIES.SYMBOL: gene_symbols,
        ONTOLOGIES.ENSEMBL_GENE: [gene_map.get(x, f'{x}_unknown_ensg') for x in gene_symbols]
    })
    
    return gene_table


def _extract_attention_weights(model) -> List[AttentionLayer]:
    """
    Extract core attention weights (Q, K, V, O) from all layers as AttentionLayer instances
    
    Parameters
    ----------
    model : AIDOCell model
        The AIDOCell model
        
    Returns
    -------
    List[AttentionLayer]
        List of AttentionLayer instances
    """
    attention_layers = []
    encoder = model.encoder
    transformer_layers = encoder.encoder.layer
    n_layers = model.get_num_layer()
    
    for layer_idx in range(n_layers):
        layer = transformer_layers[layer_idx]
        attention_self = layer.attention.self
        attention_output = layer.attention.output
        
        attention_layers.append(
            AttentionLayer(
                layer_idx=layer_idx,
                W_q=attention_self.query.weight.detach().cpu().numpy(),
                W_k=attention_self.key.weight.detach().cpu().numpy(),
                W_v=attention_self.value.weight.detach().cpu().numpy(),
                W_o=attention_output.dense.weight.detach().cpu().numpy(),
            )
        )
    
    return attention_layers


def extract_model_weights(model) -> FoundationModelWeights:
    """
    Extract model weights in the standardized format
    
    Parameters
    ----------
    model : AIDOCell model
        The AIDOCell model
        
    Returns
    -------
    FoundationModelWeights
        FoundationModelWeights instance containing gene_embedding and attention_layers
    """
    
    # Extract gene embeddings
    encoder = model.encoder
    n_genes = len(load_gene_annotations())
    
    with torch.no_grad():
        gene_positions = torch.arange(n_genes)
        gene_embedding = encoder.position_embedding(gene_positions).cpu().numpy()
    
    # Extract attention weights as AttentionLayer instances
    attention_layers = _extract_attention_weights(model)
    
    return FoundationModelWeights(
        gene_embedding=gene_embedding,
        attention_layers=attention_layers,
    )


def _format_model_metadata(model, model_class_name):
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
    vocab_list = gene_annotations[FM_DEFS.VOCAB_NAME].tolist()
    
    return {
        FM_DEFS.MODEL_NAME: AIDOCELL_DEFS.MODEL_NAME,
        FM_DEFS.MODEL_VARIANT: model_class_name,
        FM_DEFS.N_GENES: n_genes,
        FM_DEFS.N_VOCAB: n_genes,  # Same as n_genes for AIDOCell (no special tokens)
        FM_DEFS.ORDERED_VOCABULARY: vocab_list,
        FM_DEFS.EMBED_DIM: int(model.get_embedding_size()),
        FM_DEFS.N_LAYERS: int(model.get_num_layer()),
        FM_DEFS.N_HEADS: int(encoder.config.num_attention_heads),
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
    model_metadata = _format_model_metadata(model, model_class_name = model_class.__name__)
    
    return model, gene_annotations, model_metadata
