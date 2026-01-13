""" Utility functions for interacting with the scFoundation model. """

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from napistu.constants import ONTOLOGIES
from napistu_torch.load.constants import FM_DEFS, FOUNDATION_MODEL_NAMES
from napistu_torch.load.foundation_models import (
    AttentionLayer,
    FoundationModel,
    FoundationModelWeights,
)

from AIDOCell import load_gene_annotations

logger = logging.getLogger(__name__)


REPO_ID = "genbio-ai/scFoundation"
CHECKPOINT_FILE = "models.ckpt"
GENE_LIST_URL = "https://raw.githubusercontent.com/biomap-research/scFoundation/main/OS_scRNA_gene_index.19264.tsv"

N_GENES = 19264
EMBED_DIM = 768
N_ENCODER_LAYERS = 12
N_HEADS = 12

GENE_ENCOER = "gene"


def process_scfoundation(
    checkpoint_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_prefix: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> FoundationModel:
    """
    Process scFoundation checkpoint and optionally save to disk.
    
    Parameters
    ----------
    checkpoint_path : str, optional
        Path to local checkpoint. If None, downloads from HuggingFace.
    output_dir : str, optional
        Directory to save processed model. If None, doesn't save.
    output_prefix : str, optional
        Prefix for output files (default: "scFoundation_{variant}")
    cache_dir : str, optional
        Cache directory for HuggingFace downloads
        
    Returns
    -------
    FoundationModel
        Processed model instance
        
    Examples
    --------
    >>> # Download and process (don't save)
    >>> model = process_scfoundation()
    >>>
    >>> # Download and save
    >>> model = process_scfoundation(output_dir="./models")
    >>>
    >>> # Process local file
    >>> model = process_scfoundation(
    ...     checkpoint_path="./models.ckpt",
    ...     output_dir="./models"
    ... )
    """
    
    # Download checkpoint if needed
    if checkpoint_path is None:
        logger.info(f"Downloading scFoundation checkpoint from HuggingFace...")
        checkpoint_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=CHECKPOINT_FILE,
            cache_dir=cache_dir,
        )
    
    # Load checkpoint
    logger.info(f"Loading scFoundation checkpoint (gene encoder: {GENE_ENCOER})")
    full_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if GENE_ENCOER not in full_checkpoint:
        raise ValueError(
            f"Gene encoder '{GENE_ENCOER}' not found in checkpoint. "
            f"Available: {list(full_checkpoint.keys())}"
        )
    
    checkpoint = full_checkpoint[GENE_ENCOER]
    
    # Extract components
    gene_annotations = load_gene_annotations()
    weights = extract_weights(checkpoint)
    metadata = extract_metadata(checkpoint, gene_annotations)
    
    # Build model
    model = FoundationModel(
        weights=weights,
        gene_annotations=gene_annotations,
        model_metadata=metadata,
    )
    
    # Save if requested
    if output_dir is not None:
        if output_prefix is None:
            output_prefix = FOUNDATION_MODEL_NAMES.SCFOUNDATION
        logger.info(f"Saving to {output_dir}/{output_prefix}")
        model.save(output_dir, output_prefix)
    
    return model


# ============================================================================
# Weight Extraction
# ============================================================================


def extract_weights(checkpoint: dict) -> FoundationModelWeights:
    """
    Extract gene embeddings and attention weights from checkpoint.
    
    Parameters
    ----------
    checkpoint : dict
        Loaded checkpoint for specific variant
        
    Returns
    -------
    FoundationModelWeights
        Extracted weights in standard format
    """
    logger.info("Extracting model weights...")
    state_dict = checkpoint["state_dict"]
    
    # Gene embeddings (exclude special tokens)
    gene_emb_full = state_dict["model.pos_emb.weight"].cpu().numpy()
    gene_embedding = gene_emb_full[:N_GENES, :]
    
    logger.info(f"Extracted gene embeddings: {gene_embedding.shape}")
    
    # Attention layers
    attention_layers = []
    for layer_idx in range(N_ENCODER_LAYERS):
        # Combined QKV projection
        in_proj_key = f"model.encoder.transformer_encoder.{layer_idx}.self_attn.in_proj_weight"
        out_proj_key = f"model.encoder.transformer_encoder.{layer_idx}.self_attn.out_proj.weight"
        
        in_proj = state_dict[in_proj_key].cpu().numpy()
        out_proj = state_dict[out_proj_key].cpu().numpy()
        
        # Split combined QKV (shape: [3*embed_dim, embed_dim])
        w_q = in_proj[:EMBED_DIM, :]
        w_k = in_proj[EMBED_DIM : 2 * EMBED_DIM, :]
        w_v = in_proj[2 * EMBED_DIM :, :]
        
        attention_layers.append(
            AttentionLayer(
                layer_idx=layer_idx,
                W_q=w_q,
                W_k=w_k,
                W_v=w_v,
                W_o=out_proj,
            )
        )
    
    logger.info(f"Extracted {len(attention_layers)} attention layers")
    
    return FoundationModelWeights(
        gene_embedding=gene_embedding,
        attention_layers=attention_layers,
    )


# ============================================================================
# Metadata Extraction
# ============================================================================


def extract_metadata(
    checkpoint: dict, 
    gene_annotations: pd.DataFrame, 
) -> dict:
    """
    Extract model metadata from checkpoint config.
    
    Parameters
    ----------
    checkpoint : dict
        Loaded checkpoint for gene encoder
    gene_annotations : pd.DataFrame
        Gene annotations table
        
    Returns
    -------
    dict
        Metadata dictionary for FoundationModel
    """
    logger.info("Extracting metadata...")
    
    config = checkpoint["config"]
    encoder_config = config["model_config"]["mae_autobin"]["encoder"]
    
    n_genes = len(gene_annotations)
    vocab_list = gene_annotations[FM_DEFS.VOCAB_NAME].tolist()
    
    return {
        FM_DEFS.MODEL_NAME: FOUNDATION_MODEL_NAMES.SCFOUNDATION,
        FM_DEFS.N_GENES: n_genes,
        FM_DEFS.N_VOCAB: n_genes,
        FM_DEFS.ORDERED_VOCABULARY: vocab_list,
        FM_DEFS.EMBED_DIM: encoder_config["hidden_dim"],
        FM_DEFS.N_LAYERS: encoder_config["depth"],
        FM_DEFS.N_HEADS: encoder_config["heads"],
    }

