""" Common cross-model utility functions. """

import os
import json
import logging
from types import SimpleNamespace
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy.special import softmax

logger = logging.getLogger(__name__)


# this matches the Napistu namespace but i don't want to actually add a dependency since Napistu requires Python 3.11+
# and some model still use 3.10
ONTOLOGIES = SimpleNamespace(
    ENSEMBL_GENE = "ensembl_gene",
    SYMBOL = "symbol",
)

RESULTS_DEFS = SimpleNamespace(
    # model summaries
    GENE_EMBEDDING = "gene_embedding",
    LAYER_NAME_TEMPLATE = "layer_{layer_idx}",
    ATTENTION_WEIGHTS = "attention_weights",
    W_Q = "W_q",
    W_K = "W_k",
    W_V = "W_v",
    W_O = "W_o",
    # gene metadata
    GENE_ANNOTATIONS = "gene_annotations",
    VOCAB_NAME = "vocab_name",
    # model metadata
    MODEL_METADATA = "model_metadata",
    MODEL_NAME = "model_name",
    N_GENES = "n_genes",
    N_VOCAB = "n_vocab",
    ORDERED_VOCABULARY = "ordered_vocabulary",
    EMBED_DIM = "embed_dim",
    N_LAYERS = "n_layers",
    N_HEADS = "n_heads",
)

def save_results(weights_dict, gene_annotations, model_metadata, output_dir, output_prefix):
    """
    Save foundation model results to files.
    
    Parameters
    ----------
    weights_dict : dict
        Dictionary containing gene_embedding and attention_weights numpy arrays
    gene_annotations : pandas.DataFrame
        DataFrame with gene annotations containing gene_index and ensembl_gene columns
    model_metadata : dict
        Dictionary with model metadata (model_name, n_genes, embed_dim, n_layers, n_heads)
    output_dir : str
        Directory path to save files
    output_prefix : str
        Prefix for output filenames (will create {prefix}_weights.npz and {prefix}_metadata.json)
        
    Notes
    -----
    This function validates all input data using Pydantic validators before saving.
    Creates output directory if it doesn't exist.
    """
    weights_filename, metadata_filename = _prefix_to_savefiles(output_prefix)
    weights_path = os.path.join(output_dir, weights_filename)
    metadata_path = os.path.join(output_dir, metadata_filename)

    logger.info(f"Saving weights to {weights_path} and metadata to {metadata_path}")
    
    # validating data structure
    FoundationModelData(**weights_dict)
    ModelMetadata(**model_metadata)
    GeneAnnotations(annotations=gene_annotations)

    logger.info("Successfully validated weights, gene metadata and model metadata")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save weights data to npz
    logger.info(f"Saving weights to {weights_path}")
    np.savez(weights_path, **weights_dict)
    
    # Combine gene_annotations and model_metadata into single JSON
    logger.info(f"Saving metadata to {metadata_path}")
    combined_metadata = {
        RESULTS_DEFS.MODEL_METADATA: model_metadata,
        RESULTS_DEFS.GENE_ANNOTATIONS: gene_annotations.to_dict('records')  # Convert DataFrame to list of dicts
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(combined_metadata, f, indent=2)
    
    logger.info("Successfully saved all results")


def load_results(output_dir, output_prefix) -> Tuple[dict, pd.DataFrame, dict]:
    """
    Load foundation model results from files.
    
    Parameters
    ----------
    output_dir : str
        Directory path containing the saved files
    output_prefix : str
        Prefix used for the saved files (will load {prefix}_weights.npz and {prefix}_metadata.json)
        
    Returns
    -------
    weights_dict : dict
        Dictionary containing gene_embedding and attention_weights numpy arrays
    gene_annotations : pandas.DataFrame
        DataFrame with gene annotations containing gene_index and ensembl_gene columns
    model_metadata : dict
        Dictionary with model metadata (model_name, n_genes, embed_dim, n_layers, n_heads)
        
    Notes
    -----
    This function validates all loaded data using Pydantic validators.
    Raises validation errors if the loaded data doesn't match expected structure.
    """
    weights_filename, metadata_filename = _prefix_to_savefiles(output_prefix)
    weights_path = os.path.join(output_dir, weights_filename)
    metadata_path = os.path.join(output_dir, metadata_filename)
    
    logger.info(f"Loading weights from {weights_path} and metadata from {metadata_path}")
    
    # Load weights from npz
    logger.info(f"Loading weights from {weights_path}")
    weights_data = np.load(weights_path, allow_pickle=True)
    weights_dict = {}
    for key in weights_data.keys():
        value = weights_data[key]
        # Handle numpy arrays containing objects (like dictionaries)
        if isinstance(value, np.ndarray) and value.dtype == object:
            weights_dict[key] = value.item()
        else:
            weights_dict[key] = value
    
    # Load metadata from JSON
    logger.info(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r') as f:
        combined_metadata = json.load(f)
    
    model_metadata = combined_metadata[RESULTS_DEFS.MODEL_METADATA]
    gene_annotations = pd.DataFrame(combined_metadata[RESULTS_DEFS.GENE_ANNOTATIONS])
    
    # Validate loaded data
    FoundationModelData(**weights_dict)
    ModelMetadata(**model_metadata)
    GeneAnnotations(annotations=gene_annotations)
    
    logger.info("Successfully loaded and validated all results")
    
    return weights_dict, gene_annotations, model_metadata


def compute_attention_from_weights(embeddings, W_q, W_k):
    """
    Compute attention scores from embeddings and weight matrices.
    
    Parameters
    ----------
    embeddings : numpy.ndarray
        Gene embeddings matrix of shape (n_genes, embed_dim)
    W_q : numpy.ndarray
        Query weight matrix of shape (embed_dim, d_k)
    W_k : numpy.ndarray
        Key weight matrix of shape (embed_dim, d_k)
        
    Returns
    -------
    numpy.ndarray
        Attention scores matrix of shape (n_genes, n_genes) with softmax applied
        
    Notes
    -----
    Computes scaled dot-product attention: Attention(Q,K) = softmax(QK^T / sqrt(d_k))
    where Q = embeddings @ W_q.T and K = embeddings @ W_k.T
    """
    Q = embeddings @ W_q.T
    K = embeddings @ W_k.T
    attn_scores = (Q @ K.T) / np.sqrt(Q.shape[-1])
    return softmax(attn_scores, axis=-1)


class FoundationModelData(BaseModel):
    """Simple validator for foundation model data dictionary structure."""
    
    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}
    
    gene_embedding: np.ndarray = Field(..., alias=RESULTS_DEFS.GENE_EMBEDDING)
    attention_weights: Dict[str, Dict[str, np.ndarray]] = Field(..., alias=RESULTS_DEFS.ATTENTION_WEIGHTS)
    
    @field_validator('gene_embedding')
    @classmethod
    def validate_gene_embedding(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError("gene_embedding must be a numpy array")
        if v.ndim != 2:
            raise ValueError("gene_embedding must be 2-dimensional")
        return v
    
    @field_validator('attention_weights')
    @classmethod
    def validate_attention_weights_structure(cls, v):
        if not isinstance(v, dict):
            raise ValueError("attention_weights must be a dictionary")
        
        # Check layer structure
        for layer_name, layer_data in v.items():
            if not layer_name.startswith('layer_'):
                raise ValueError(f"Layer name must start with 'layer_', got: {layer_name}")
            
            if not isinstance(layer_data, dict):
                raise ValueError(f"Layer {layer_name} must be a dictionary")
            
            # Check required weight matrices
            required_weights = [RESULTS_DEFS.W_Q, RESULTS_DEFS.W_K, RESULTS_DEFS.W_V, RESULTS_DEFS.W_O]
            for weight_name in required_weights:
                if weight_name not in layer_data:
                    raise ValueError(f"Layer {layer_name} missing required weight matrix: {weight_name}")
                
                weight_matrix = layer_data[weight_name]
                if not isinstance(weight_matrix, np.ndarray):
                    raise ValueError(f"Weight matrix {weight_name} in {layer_name} must be a numpy array")
                if weight_matrix.ndim != 2:
                    raise ValueError(f"Weight matrix {weight_name} in {layer_name} must be 2-dimensional")
        
        return v
    
    @model_validator(mode='after')
    def validate_embedding_attention_consistency(self):
        """Validate that embedding dimensions are consistent with attention weights."""
        embed_dim = self.gene_embedding.shape[1]
        
        # Check that all attention weight matrices have consistent dimensions
        for layer_name, layer_data in self.attention_weights.items():
            for weight_name, weight_matrix in layer_data.items():
                if weight_matrix.shape[0] != embed_dim:
                    raise ValueError(
                        f"Attention weight {weight_name} in {layer_name} has "
                        f"inconsistent dimension: expected {embed_dim}, got {weight_matrix.shape[0]}"
                    )
        
        return self


class ModelMetadata(BaseModel):
    """Simple validator for model metadata dictionary structure."""
    
    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}
    
    model_name: str = Field(..., alias=RESULTS_DEFS.MODEL_NAME)
    n_genes: int = Field(..., alias=RESULTS_DEFS.N_GENES)
    n_vocab: int = Field(..., alias=RESULTS_DEFS.N_VOCAB)
    ordered_vocabulary: list = Field(..., alias=RESULTS_DEFS.ORDERED_VOCABULARY, 
                                   description="Vocabulary terms in same order as embedding matrix rows (index i corresponds to embedding row i)")
    embed_dim: int = Field(..., alias=RESULTS_DEFS.EMBED_DIM)
    n_layers: int = Field(..., alias=RESULTS_DEFS.N_LAYERS)
    n_heads: int = Field(..., alias=RESULTS_DEFS.N_HEADS)
    
    @field_validator('n_genes', 'n_vocab', 'embed_dim', 'n_layers', 'n_heads')
    @classmethod
    def validate_positive_integers(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"Value must be a positive integer, got: {v}")
        return v
    
    @field_validator('ordered_vocabulary')
    @classmethod
    def validate_ordered_vocabulary(cls, v):
        if not isinstance(v, list):
            raise ValueError("ordered_vocabulary must be a list")
        if not all(isinstance(item, str) for item in v):
            raise ValueError("ordered_vocabulary must contain only strings")
        return v
    
    @model_validator(mode='after')
    def validate_vocab_gene_relationship(self):
        """Validate that n_vocab >= n_genes and matches ordered_vocabulary length"""
        if self.n_vocab < self.n_genes:
            raise ValueError(f"n_vocab ({self.n_vocab}) must be >= n_genes ({self.n_genes})")
        if len(self.ordered_vocabulary) != self.n_vocab:
            raise ValueError(f"ordered_vocabulary length ({len(self.ordered_vocabulary)}) must match n_vocab ({self.n_vocab})")
        return self


class GeneAnnotations(BaseModel):
    """Simple validator for gene annotations DataFrame structure."""
    
    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}
    
    annotations: pd.DataFrame = Field(...)
    
    @field_validator('annotations')
    @classmethod
    def validate_annotations_structure(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError("annotations must be a pandas DataFrame")
        
        # Check required columns
        required_columns = [RESULTS_DEFS.VOCAB_NAME, ONTOLOGIES.ENSEMBL_GENE]
        for col in required_columns:
            if col not in v.columns:
                raise ValueError(f"DataFrame missing required column: {col}")
        
        # Validate vocab_name column
        if not pd.api.types.is_string_dtype(v[RESULTS_DEFS.VOCAB_NAME]):
            raise ValueError(f"Column {RESULTS_DEFS.VOCAB_NAME} must contain strings")
        
        # Check for unique vocab_name values
        if v[RESULTS_DEFS.VOCAB_NAME].duplicated().any():
            raise ValueError(f"Column {RESULTS_DEFS.VOCAB_NAME} must contain unique values")
        
        # Check for missing vocab_name values
        if v[RESULTS_DEFS.VOCAB_NAME].isna().any():
            raise ValueError(f"Column {RESULTS_DEFS.VOCAB_NAME} must not contain missing values")
        
        # Validate ensembl_gene column
        if not pd.api.types.is_string_dtype(v[ONTOLOGIES.ENSEMBL_GENE]):
            raise ValueError(f"Column {ONTOLOGIES.ENSEMBL_GENE} must contain strings")
        
        return v

def _prefix_to_savefiles(prefix: str) -> Tuple[str, str]:
    """
    Generate filenames for weights and metadata files from a prefix.
    
    Parameters
    ----------
    prefix : str
        Base prefix for the filenames
        
    Returns
    -------
    weights_filename : str
        Filename for weights file: "{prefix}_weights.npz"
    metadata_filename : str
        Filename for metadata file: "{prefix}_metadata.json"
        
    Examples
    --------
    >>> _prefix_to_savefiles("scgpt")
    ("scgpt_weights.npz", "scgpt_metadata.json")
    """
    return f"{prefix}_weights.npz", f"{prefix}_metadata.json"
