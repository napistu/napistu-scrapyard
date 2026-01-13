"""Common utilities for ETLing foundation models."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from napistu_torch.load.constants import FM_DEFS
from napistu_torch.load.foundation_models import (
    AttentionLayer,
    FoundationModel,
    FoundationModelWeights,
)

logger = logging.getLogger(__name__)


def split_qkv_weights(
    qkv_weight: np.ndarray, embed_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Split combined QKV weight matrix into separate Q, K, V matrices.

  Parameters
  ----------
  qkv_weight : np.ndarray
    Combined QKV weight matrix of shape (3 * embed_dim, embed_dim)
  embed_dim : int
    Embedding dimension

  Returns
  -------
  Tuple[np.ndarray, np.ndarray, np.ndarray]
    Tuple of (W_q, W_k, W_v) matrices, each of shape (embed_dim, embed_dim)

  Examples
  --------
  >>> qkv = np.random.randn(768, 256)  # 3*256 = 768
  >>> w_q, w_k, w_v = split_qkv_weights(qkv, embed_dim=256)
  >>> assert w_q.shape == (256, 256)
  """
  if qkv_weight.shape[0] != 3 * embed_dim:
    raise ValueError(
      f"Expected qkv_weight.shape[0] to be 3*embed_dim ({3*embed_dim}), "
      f"but got {qkv_weight.shape[0]}"
    )

  w_q = qkv_weight[:embed_dim, :]
  w_k = qkv_weight[embed_dim : 2 * embed_dim, :]
  w_v = qkv_weight[2 * embed_dim :, :]

  return w_q, w_k, w_v


def create_and_save_foundation_model(
    weights: FoundationModelWeights,
    gene_annotations: pd.DataFrame,
    model_metadata: Dict,
    output_dir: str,
    file_prefix: str,
) -> FoundationModel:
  """
  Create FoundationModel instance and save to disk.

  Parameters
  ----------
  weights : FoundationModelWeights
    Model weights
  gene_annotations : pd.DataFrame
    Gene annotations DataFrame
  model_metadata : Dict
    Model metadata dictionary
  output_dir : str
    Output directory for saving
  file_prefix : str
    Prefix for output files

  Returns
  -------
  FoundationModel
    Created FoundationModel instance

  Examples
  --------
  >>> model = create_and_save_foundation_model(
  ...     weights, annotations, metadata, "./output", "scGPT"
  ... )
  """
  logger.info(f"Creating FoundationModel and saving...")
  foundation_model = FoundationModel(
    weights=weights,
    gene_annotations=gene_annotations,
    model_metadata=model_metadata,
  )
  foundation_model.save(output_dir, file_prefix)
  logger.info(f"Successfully saved all results!")
  return foundation_model


def format_base_metadata(
    model_name: str,
    n_genes: int,
    n_vocab: int,
    vocab_list: List[str],
    embed_dim: int,
    n_layers: int,
    n_heads: int,
    model_variant: Optional[str] = None,
    **extra_metadata,
) -> Dict:
  """
  Format base model metadata dictionary with standard keys.

  Parameters
  ----------
  model_name : str
    Model name (e.g., "scGPT", "scFoundation")
  n_genes : int
    Number of genes
  n_vocab : int
    Vocabulary size (may include special tokens)
  vocab_list : List[str]
    Ordered vocabulary list
  embed_dim : int
    Embedding dimension
  n_layers : int
    Number of transformer layers
  n_heads : int
    Number of attention heads
  model_variant : str, optional
    Model variant identifier (e.g., "small", "medium")
  **extra_metadata : Dict
    Additional metadata to include

  Returns
  -------
  Dict
    Metadata dictionary with standard FM_DEFS keys

  Examples
  --------
  >>> metadata = format_base_metadata(
  ...     "scGPT", 1000, 1003, ["gene1", "gene2", ...], 512, 12, 8
  ... )
  """
  metadata = {
    FM_DEFS.MODEL_NAME: model_name,
    FM_DEFS.N_GENES: n_genes,
    FM_DEFS.N_VOCAB: n_vocab,
    FM_DEFS.ORDERED_VOCABULARY: vocab_list,
    FM_DEFS.EMBED_DIM: embed_dim,
    FM_DEFS.N_LAYERS: n_layers,
    FM_DEFS.N_HEADS: n_heads,
  }

  if model_variant is not None:
    metadata[FM_DEFS.MODEL_VARIANT] = model_variant

  # Add any extra metadata
  metadata.update(extra_metadata)

  return metadata


def extract_attention_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    n_layers: int,
    embed_dim: int,
    layer_prefix: str = "transformer_encoder.layers",
    qkv_key_template: str = "{layer_idx}.self_attn.Wqkv.weight",
    out_proj_key_template: str = "{layer_idx}.self_attn.out_proj.weight",
) -> List[AttentionLayer]:
  """
  Extract attention layers from a state dict using configurable key templates.

  Parameters
  ----------
  state_dict : Dict[str, torch.Tensor]
    Model state dictionary
  n_layers : int
    Number of transformer layers
  embed_dim : int
    Embedding dimension
  layer_prefix : str
    Prefix for layer keys (default: "transformer_encoder.layers")
  qkv_key_template : str
    Template for QKV weight key, use {layer_idx} placeholder
    (default: "{layer_idx}.self_attn.Wqkv.weight")
  out_proj_key_template : str
    Template for output projection key, use {layer_idx} placeholder
    (default: "{layer_idx}.self_attn.out_proj.weight")

  Returns
  -------
  List[AttentionLayer]
    List of AttentionLayer instances

  Examples
  --------
  >>> layers = extract_attention_from_state_dict(
  ...     state_dict, n_layers=12, embed_dim=512
  ... )
  """
  attention_layers = []

  for layer_idx in range(n_layers):
    # Format keys with layer index
    qkv_key = f"{layer_prefix}.{qkv_key_template.format(layer_idx=layer_idx)}"
    out_proj_key = (
      f"{layer_prefix}.{out_proj_key_template.format(layer_idx=layer_idx)}"
    )

    if qkv_key not in state_dict:
      raise KeyError(f"Could not find {qkv_key} in state_dict")
    if out_proj_key not in state_dict:
      raise KeyError(f"Could not find {out_proj_key} in state_dict")

    # Extract and convert to numpy
    qkv_weight = state_dict[qkv_key].clone().cpu().detach().numpy()
    out_proj = state_dict[out_proj_key].clone().cpu().detach().numpy()

    # Split QKV
    w_q, w_k, w_v = split_qkv_weights(qkv_weight, embed_dim)

    attention_layers.append(
      AttentionLayer(
        layer_idx=layer_idx, W_q=w_q, W_k=w_k, W_v=w_v, W_o=out_proj
      )
    )

  return attention_layers
