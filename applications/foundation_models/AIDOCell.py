import json
import numpy as np
import pandas as pd
import torch
import os

import modelgenerator.cell.utils as cell_utils

def load_model(model_class):
    """Load and return model in eval mode
    
    Args:
        model_class: Model class to load
    """
    
    model = model_class(
        legacy_adapter_type=None,
        default_config=None,
        from_scratch=False
    )
    model.eval()
    return model


def create_gene_mapping_table():
    """Create a comprehensive gene mapping table with Ensembl IDs"""
    
    load_base = os.path.dirname(os.path.abspath(cell_utils.__file__))
    gene_file = os.path.join(load_base, 'gene_lists/OS_scRNA_gene_index.19264.tsv')
    
    # Load gene symbols
    gene_symbols = pd.read_csv(gene_file, sep='\t')['gene_name'].values
    
    # Build the mapping from symbols to Ensembl IDs (this is what the function does)
    gene_map = cell_utils.build_map(gene_symbols)
    
    # Create the mapping table
    gene_table = pd.DataFrame({
        'gene_index': np.arange(len(gene_symbols)),
        'gene_symbol': gene_symbols,
        'ensembl_id': [gene_map.get(x, f'{x}_unknown_ensg') for x in gene_symbols]
    })
    
    # Flag unmapped genes
    gene_table['is_mapped'] = ~gene_table['ensembl_id'].str.endswith('_unknown_ensg')
    
    return gene_table


def extract_attention_weights(model, n_layers):
    """
    Extract core attention weights (Q, K, V, O) from all layers
    
    Returns:
        dict: {layer_0: {W_q, W_k, W_v, W_o}, layer_1: {...}, ...}
    """
    attention_weights = {}
    encoder = model.encoder
    transformer_layers = encoder.encoder.layer
    
    for layer_idx in range(n_layers):
        layer = transformer_layers[layer_idx]
        attention_self = layer.attention.self
        attention_output = layer.attention.output
        
        attention_weights[f'layer_{layer_idx}'] = {
            'W_q': attention_self.query.weight.detach().cpu().numpy(),
            'W_k': attention_self.key.weight.detach().cpu().numpy(),
            'W_v': attention_self.value.weight.detach().cpu().numpy(),
            'W_o': attention_output.dense.weight.detach().cpu().numpy(),
        }
    
    return attention_weights


def extract_gene_embeddings(model, n_genes):
    """
    Extract position embeddings for genes
    
    Returns:
        np.ndarray: (n_genes, embed_dim)
    """
    encoder = model.encoder
    
    with torch.no_grad():
        gene_positions = torch.arange(n_genes)
        embeddings = encoder.position_embedding(gene_positions).cpu().numpy()
    
    return embeddings


def extract_model_metadata(model, model_name, n_genes):
    """
    Extract model architecture metadata
    
    Returns:
        dict: Model configuration information
    """
    encoder = model.encoder
    
    return {
        'model_name': model_name,
        'n_genes': int(n_genes),
        'embed_dim': int(model.get_embedding_size()),
        'n_layers': int(model.get_num_layer()),
        'n_heads': int(encoder.config.num_attention_heads),
        'hidden_dim': int(encoder.config.hidden_size),
    }


def format_weights_for_saving(embeddings, attention_weights):
    """
    Flatten nested attention weights dict for npz format
    
    Args:
        embeddings: np.ndarray of gene embeddings
        attention_weights: dict of {layer_X: {W_q, W_k, W_v, W_o}}
    
    Returns:
        dict: Flattened dictionary ready for np.savez
    """
    weights_dict = {'embeddings': embeddings}
    
    # Flatten attention weights: layer_0_W_q, layer_0_W_k, etc.
    for layer_name, layer_weights in attention_weights.items():
        for weight_name, weight_array in layer_weights.items():
            weights_dict[f'{layer_name}_{weight_name}'] = weight_array
    
    return weights_dict


def format_metadata_for_saving(model_metadata, gene_table, files):
    """
    Create complete metadata dict for JSON
    
    Args:
        model_metadata: dict from extract_model_metadata()
        gene_table: pd.DataFrame with gene information
        files: dict with filenames
    
    Returns:
        dict: Complete metadata ready for JSON
    """
    return {
        'model_info': model_metadata,
        'gene_table': {
            'gene_index': gene_table['gene_index'].tolist(),
            'gene_symbol': gene_table['gene_symbol'].tolist(),
            'ensembl_id': gene_table['ensembl_id'].tolist(),
            'is_mapped': gene_table['is_mapped'].tolist(),
        },
        'files': files,
    }
