"""
Utility functions for analyzing foundation models. Once these stabalize a bit, they will be moved to the napistu-torch library.
"""

import logging

import numpy as np
import pandas as pd
import torch

# local .py file
from etl_utils import (
    load_results,
    RESULTS_DEFS,
    ONTOLOGIES,
)

logger = logging.getLogger(__name__)

# data loading

def load_foundation_summaries(model_prefixes: str, output_dir: str, verbose: bool = True) -> dict:

    """
    Loads the results for the foundation models.

    Parameters
    ----------
    model_prefixes : str
        The file name prefixes of the foundation models to load. Each model saves weights and metadata to separate files.
    output_dir : str
        The directory to load the results from.
    verbose : bool
        Extra reporting

    Returns
    -------
    models_dict : dict
        A dictionary of the foundation models.
    """

    models_dict = dict()
    for prefix in model_prefixes:
        if verbose:
            logger.info(f"Loading results for the {prefix} foundation model")
        weights_dict, gene_annotations, model_metadata = load_results(output_dir, prefix)
        models_dict[prefix] = dict()
        models_dict[prefix][RESULTS_DEFS.WEIGHTS_DICT] = weights_dict
        models_dict[prefix][RESULTS_DEFS.GENE_ANNOTATIONS] = gene_annotations
        models_dict[prefix][RESULTS_DEFS.MODEL_METADATA] = model_metadata

    return models_dict


def get_common_identifiers(models_dict: dict, ontology: str = ONTOLOGIES.ENSEMBL_GENE, verbose: bool = True) -> list[str]:

    """
    Loads the common identifiers across all models.

    Parameters
    ----------
    models_dict : dict
        A dictionary of the foundation models.
    ontology : str
        The ontology to use for the common identifiers. This should be a column in every model's gene annotations.
    verbose : bool
        Extra reporting

    Returns
    -------
    common_identifiers : list[str]
        A list of the common identifiers across all models.
    """

    # Get common identifiers across all models
    common_identifiers = None
    for model_data in models_dict.values():

        if ontology not in model_data[RESULTS_DEFS.GENE_ANNOTATIONS].columns:
            raise ValueError(f"The ontology {ontology} is not a column in the gene annotations for the {model_data[RESULTS_DEFS.MODEL_METADATA][RESULTS_DEFS.MODEL_NAME]} model")

        identifiers = set(model_data[RESULTS_DEFS.GENE_ANNOTATIONS][ontology])
        if common_identifiers is None:
            common_identifiers = identifiers
        else:
            common_identifiers = common_identifiers.intersection(identifiers)

    common_identifiers = list(common_identifiers)

    if verbose:
        logger.info(f"Found {len(common_identifiers)} Ensembl gene IDs shared across all models")
    
    return common_identifiers

def get_aligned_embeddings(models_dict: dict, common_identifiers: list[str], ontology: str = ONTOLOGIES.ENSEMBL_GENE):

    """
    Aligns the gene embeddings across all models.

    This function will align the gene embeddings across all models based on the common identifiers. Embeddings are aligned by:
    1. Adding a positional index to the gene embeddings which maps each gene to a row in the embedding matrix.
    2. Filtering and reordering the gene annotations so they match the order of the common identifiers.
    3. Using the positional index to reorder the gene embeddings.

    Parameters
    ----------
    models_dict : dict
        A dictionary of the foundation models' summaries.
    common_identifiers : list[str]
        A list of the common identifiers across all models. This will define the order of rows in the aligned embeddings.
    ontology : str
        The ontology to use for the common identifiers. This should be a column in every model's gene annotations.

    Returns
    -------
    aligned_embeddings : dict
        A dictionary of the aligned embeddings for each model.
    """

    aligned_embeddings = dict()
    for prefix, model_data in models_dict.items():

        # load model-specific (meta)data        
        gene_embedding = model_data[RESULTS_DEFS.WEIGHTS_DICT][RESULTS_DEFS.GENE_EMBEDDING]
        gene_annotations = model_data[RESULTS_DEFS.GENE_ANNOTATIONS]
        if ontology not in gene_annotations.columns:
            raise ValueError(f"The ontology {ontology} is not a column in the gene annotations for the {model_data[RESULTS_DEFS.MODEL_METADATA][RESULTS_DEFS.MODEL_NAME]} model")
        ordered_vocab = model_data[RESULTS_DEFS.MODEL_METADATA][RESULTS_DEFS.ORDERED_VOCABULARY]

        vocab_df = pd.DataFrame({RESULTS_DEFS.VOCAB_NAME: ordered_vocab}).assign(index_position = range(0, len(ordered_vocab)))

        # filter to common identifiers and add the ordering in the vocab (i.e., the rows in the embedding matrix)
        embedding_alignment_lookup_table = (
            gene_annotations
            .set_index(ONTOLOGIES.ENSEMBL_GENE)
            # filter to common identifiers and reorder based on common_identifiers' ordering
            .loc[common_identifiers]
            .merge(vocab_df, on = RESULTS_DEFS.VOCAB_NAME, how = "inner")
        )

        # extract the embeddings for the common identifiers in the the order of common_identifiers
        aligned_embedding = gene_embedding[embedding_alignment_lookup_table.index_position]
        
        logger.info(f"{prefix}: Extracted a length {aligned_embedding.shape[1]} embedding for {aligned_embedding.shape[0]} common identifiers")

        aligned_embeddings[prefix] = aligned_embedding

    return aligned_embeddings

# torch utils

def select_device(mps_valid: bool = True):
    """
    Selects the device to use for the model.
    If MPS is available and mps_valid is True, use MPS.
    If CUDA is available, use CUDA.
    Otherwise, use CPU.

    Parameters
    ----------  
    mps_valid : bool
        Whether to use MPS if available.

    Returns
    -------
    device : torch.device
        The device to use for the model.
    """
    
    if mps_valid and torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def compute_cosine_distances_torch(embedding: np.ndarray, device: torch.device) -> np.ndarray:

    """
    Compute cosine distance matrix using PyTorch
    
    Parameters
    ----------
    embedding : np.ndarray
        The embedding tensor
    device : torch.device
        The device to use for the computation

    Returns
    -------
    cosine_dist : np.ndarray
        The cosine distance matrix
    """

    # convert the embedding to a tensor and move it to the device
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32, device=device)
    
    # normalize the embeddings
    embeddings_norm = torch.nn.functional.normalize(embedding_tensor, p=2, dim=1)
    # compute the cosine similarity matrix
    cosine_sim = torch.mm(embeddings_norm, embeddings_norm.t())
    # convert to cosine distance
    cosine_dist = 1 - cosine_sim

    # move back to the cpu and convert to numpy
    return cosine_dist.cpu().numpy()


def compute_spearman_correlation_torch(x: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    """
    Compute Spearman correlation using PyTorch (much faster than scipy)
    
    Parameters
    ----------
    x : array-like
        First vector (numpy array or similar)
    y : array-like
        Second vector (numpy array or similar)
    device : torch.device
        The device to use for the computation
        
    Returns
    -------
    rho : float
        Spearman correlation coefficient
    """
    
    # Convert to torch tensors if needed
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float().to(device)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float().to(device)
    
    # Convert values to ranks
    x_rank = torch.argsort(torch.argsort(x)).float()
    y_rank = torch.argsort(torch.argsort(y)).float()
    
    # Calculate Pearson correlation on ranks
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    
    correlation = (x_centered * y_centered).sum() / (
        torch.sqrt((x_centered ** 2).sum()) * torch.sqrt((y_centered ** 2).sum())
    )
    
    return correlation.item()