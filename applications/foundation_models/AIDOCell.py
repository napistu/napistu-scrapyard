""" Utilities for ETLing the AIDOCell foundation model. """

import logging
import numpy as np
import os
import pandas as pd
import torch
from types import SimpleNamespace
from typing import List

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

from etl_utils import (
    create_and_save_foundation_model,
    format_base_metadata,
)
from optional import require_modelgenerator

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


@require_modelgenerator
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
    create_and_save_foundation_model(
        weights, gene_annotations, model_metadata, output_dir, file_prefix
    )

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


@require_modelgenerator
def load_gene_annotations():
    """
    Load gene annotations from AIDOCell model

    This is a flat file which is bundled with the package
    
    Returns
    -------
    pd.DataFrame
        DataFrame with gene annotations
    """
    import modelgenerator.cell.utils as cell_utils
    
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
    
    return format_base_metadata(
        model_name=AIDOCELL_DEFS.MODEL_NAME,
        n_genes=n_genes,
        n_vocab=n_genes,  # Same as n_genes for AIDOCell (no special tokens)
        vocab_list=vocab_list,
        embed_dim=int(model.get_embedding_size()),
        n_layers=int(model.get_num_layer()),
        n_heads=int(encoder.config.num_attention_heads),
        model_variant=model_class_name,
        # Additional AIDOCell-specific metadata
        **{AIDOCELL_DEFS.HIDDEN_DIM: int(encoder.config.hidden_size)},
    )


@require_modelgenerator
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
