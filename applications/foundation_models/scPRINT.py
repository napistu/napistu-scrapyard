""" Utilities for ETLing the scPRINT foundation model. """

import logging
import os
from types import SimpleNamespace
from typing import List

import bionty as bt
import torch
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scdataloader.utils import populate_my_ontology
from scprint import scPrint

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

# Import SCPRINT_VERSIONS from constants
try:
    from napistu_torch.load.constants import SCPRINT_VERSIONS
except ImportError:
    # Fallback if not available
    SCPRINT_VERSIONS = SimpleNamespace(
        SMALL="small",
        MEDIUM="medium",
        LARGE="large",
    )

logger = logging.getLogger(__name__)

# Map version keys to checkpoint filenames
SCPRINT_CHECKPOINTS = SimpleNamespace(
    SMALL="small-v1.ckpt",
    MEDIUM="medium-v1.5.ckpt",
    LARGE="large-v1.ckpt",
)

SCPRINT_DEFS = SimpleNamespace(
    MODEL_NAME = FOUNDATION_MODEL_NAMES.SCPRINT,
    VERSIONS = SCPRINT_VERSIONS,
    CHECKPOINTS = SCPRINT_CHECKPOINTS,
    REPO_ID = "jkobject/scPRINT",
    # parameters
    D_MODEL = "d_model",
    N_LAYERS = "nlayers",
    N_HEADS = 4,  # Fixed architecture parameter
)

def process_model(version_key: str, output_dir: str, model_path: str = None) -> None:
    """
    Process a given scPRINT model version and save the results to the output directory.

    Parameters
    ----------
    version_key : str
        scPRINT version key (e.g., "SMALL", "MEDIUM", "LARGE")
    output_dir : str
        Output directory to save the results
    model_path : str, optional
        Path to directory where models are cached. If None, uses default "data/scPRINT"
        
    Returns
    -------
    None
    """
    
    if model_path is None:
        model_path = os.path.join("data", "scPRINT")
    
    # Get version ID and checkpoint filename from the version key
    version_id = getattr(SCPRINT_DEFS.VERSIONS, version_key)
    checkpoint_filename = getattr(SCPRINT_DEFS.CHECKPOINTS, version_key)
    file_prefix = f"{SCPRINT_DEFS.MODEL_NAME}_{version_id}"
    
    logger.info(f"Extracting: scPRINT {version_id} ({version_key})")
    
    # 1. Download and load model
    logger.info("\n1. Downloading/loading model if needed...")
    checkpoint_file = hf_hub_download(
        repo_id=SCPRINT_DEFS.REPO_ID,
        filename=checkpoint_filename,
        cache_dir=model_path
    )
    
    logger.info("Loading scPRINT model")
    model, gene_annotations, model_metadata = load_scprint(checkpoint_file, version=version_id)
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
        FM_DEFS.VOCAB_NAME: model.genes,
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


def extract_model_weights(model) -> FoundationModelWeights:
    """
    Extract model weights in the standardized format
    
    Parameters
    ----------
    model : scPrint
        The scPRINT model
        
    Returns
    -------
    FoundationModelWeights
        FoundationModelWeights instance containing gene_embedding and attention_layers
    """
    
    # Extract gene embeddings
    gene_embedding = model.gene_encoder.embeddings.weight.detach().cpu().numpy()
    
    # Extract attention weights as AttentionLayer instances
    attention_layers = _extract_attention_weights(model)
    
    return FoundationModelWeights(
        gene_embedding=gene_embedding,
        attention_layers=attention_layers,
    )


def load_scprint(checkpoint_path, transformer="normal", version: str = None):
    """
    Load the scPRINT model and return model, gene annotations, and metadata
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the scPRINT checkpoint file
    transformer : str, optional
        Transformer type, by default "normal"
    version : str, optional
        Version string (e.g., "small-v1", "medium-v1.5", "large-v1")
        
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
    model_metadata = _format_model_metadata(model, version=version)
    
    return model, gene_annotations, model_metadata

# private utils

def _extract_attention_weights(model) -> List[AttentionLayer]:
    """Extract self-attention weights (Q, K, V, O) from all layers as AttentionLayer instances"""
    attention_layers = []
    d_model = model.d_model
    n_layers = model.nlayers
    
    for layer_idx in range(n_layers):
        block = model.transformer.blocks[layer_idx]
        mixer = block.mixer
        
        # Get combined QKV weight: (3 * d_model, d_model)
        qkv_weight = mixer.Wqkv.weight.detach().cpu().numpy()
        
        attention_layers.append(
            AttentionLayer(
                layer_idx=layer_idx,
                W_q=qkv_weight[:d_model, :],
                W_k=qkv_weight[d_model:2*d_model, :],
                W_v=qkv_weight[2*d_model:, :],
                W_o=mixer.out_proj.weight.detach().cpu().numpy(),
            )
        )
    
    return attention_layers


def _format_model_metadata(model, version: str = None):
    """
    Extract model architecture metadata
    
    Parameters
    ----------
    model : scPrint
        The scPRINT model
    version : str, optional
        Version string (e.g., "small-v1", "medium-v1.5", "large-v1")
        
    Returns
    -------
    dict
        Dictionary with model metadata
    """
    
    # Get vocabulary as list of genes (scPRINT doesn't have special tokens)
    vocab_list = list(model.genes)
    n_genes = len(vocab_list)
    
    metadata = {
        FM_DEFS.MODEL_NAME: SCPRINT_DEFS.MODEL_NAME,
        FM_DEFS.N_GENES: n_genes,
        FM_DEFS.N_VOCAB: n_genes,  # Same as n_genes for scPRINT (no special tokens)
        FM_DEFS.ORDERED_VOCABULARY: vocab_list,
        FM_DEFS.EMBED_DIM: int(model.d_model),
        FM_DEFS.N_LAYERS: int(model.nlayers),
        FM_DEFS.N_HEADS: SCPRINT_DEFS.N_HEADS
    }
    
    # Add version if provided
    if version is not None:
        metadata[FM_DEFS.MODEL_VARIANT] = version
    
    return metadata