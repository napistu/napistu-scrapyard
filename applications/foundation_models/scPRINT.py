""" Utilities for ETLing the scPRINT foundation model. """

import logging

import bionty as bt
import torch
import numpy as np
import pandas as pd
from scdataloader.utils import populate_my_ontology
from scprint import scPrint

logger = logging.getLogger(__name__)

def load_scprint_model(checkpoint_path, transformer="normal"):
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


def create_gene_mapping_table(model):
    """Create gene mapping table with optional gene symbols from lamindb"""
    gene_table = pd.DataFrame({
        'gene_index': np.arange(len(model.genes)),
        'ensembl_id': model.genes,
    })
    
    # Optionally add gene symbols from lamindb
    try:
        all_genes_df = bt.Gene.filter().df()
        ensembl_to_symbol = all_genes_df.set_index('ensembl_gene_id')['symbol'].to_dict()
        gene_table['gene_symbol'] = gene_table['ensembl_id'].map(ensembl_to_symbol)
    except:
        gene_table['gene_symbol'] = gene_table['ensembl_id']
    
    return gene_table


def extract_attention_weights(model, n_layers):
    """Extract self-attention weights (Q, K, V, O) from all layers"""
    attention_weights = {}
    d_model = model.d_model
    
    for layer_idx in range(n_layers):
        block = model.transformer.blocks[layer_idx]
        mixer = block.mixer
        
        # Get combined QKV weight: (3 * d_model, d_model)
        qkv_weight = mixer.Wqkv.weight.detach().cpu().numpy()
        
        attention_weights[f'layer_{layer_idx}'] = {
            'W_q': qkv_weight[:d_model, :],
            'W_k': qkv_weight[d_model:2*d_model, :],
            'W_v': qkv_weight[2*d_model:, :],
            'W_o': mixer.out_proj.weight.detach().cpu().numpy(),
        }
    
    return attention_weights


def populate_lamin_db():
    # quick check to see if the lamin database is already configured
    organisms = bt.Organism.filter().df()
    human_defined = 'NCBITaxon:9606' in organisms['ontology_id'].values if len(organisms) > 0 else False
    if not human_defined:
        logger.info("Populating the full metadata catalog recommended by the scPRINT developers")
        # populate the full metadata catalog recommended by the scPRINT developers
        populate_my_ontology() 
    else:
        logger.info("Lamin database already configured")


def extract_gene_embeddings(model):
    """Extract gene embeddings"""
    embeddings = model.gene_encoder.embeddings.weight.detach().cpu().numpy()
    return embeddings


def extract_model_metadata(model, model_name, n_genes):
    """Extract model architecture metadata"""
    return {
        'model_name': model_name,
        'n_genes': int(n_genes),
        'embed_dim': int(model.d_model),
        'n_layers': int(model.nlayers),
        'n_heads': 4,  # From the architecture printout
        'normalization': model.normalization,
        'organisms': model.organisms,
    }


def format_weights_for_saving(embeddings, attention_weights):
    """Flatten nested attention weights dict for npz format"""
    weights_dict = {'embeddings': embeddings}
    
    for layer_name, layer_weights in attention_weights.items():
        for weight_name, weight_array in layer_weights.items():
            weights_dict[f'{layer_name}_{weight_name}'] = weight_array
    
    return weights_dict