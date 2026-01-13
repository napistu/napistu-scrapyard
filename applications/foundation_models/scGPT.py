""" Utility functions for interacting with the scGPT model. """

import logging
from typing import Tuple, List, Optional
from types import SimpleNamespace

import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.utils import set_seed
from torchtext.vocab import Vocab

# Import from napistu_torch
from napistu.constants import ONTOLOGIES
from napistu.utils import download_wget
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

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

set_seed(42)
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
N_HVG = 1200
N_BINS = 51
MASK_VALUE = -1
PAD_VALUE = -2
N_INPUT_BINS = N_BINS

SCGPT_DEFS = SimpleNamespace(
    MODEL_NAME = FOUNDATION_MODEL_NAMES.SCGPT,
    # urls
    GENE_IDENTIFIERS_URL = "https://github.com/bowang-lab/scGPT/files/13243634/gene_info.csv",
    # files
    CONFIG_FILENAME = "args.json",
    MODEL_FILENAME = "best_model.pt",
    VOCAB_FILENAME = "vocab.json",
    # parameters
    EMBSIZE = "embsize",
    NHEAD = "nheads",
    D_HID = "d_hid",
    NLAYERS = "nlayers",
    N_LAYERS_CLS = "n_layers_cls",
)


def process_scgpt(model_dir: str, output_dir: str, annotations_path: Optional[str] = None) -> None:
    """
    Process the scGPT model and save the results to the output directory.

    Parameters
    ----------
    model_dir : str
        Directory containing the scGPT model files (args.json, best_model.pt, vocab.json)
    output_dir : str
        Output directory to save the results
    annotations_path : str, optional
        Path to gene annotations file. If None, downloads from GENE_IDENTIFIERS_URL
        
    Returns
    -------
    None
    """
    
    file_prefix = SCGPT_DEFS.MODEL_NAME
    
    logger.info(f"Extracting: scGPT")
    
    # 1. Download and load gene annotations
    logger.info("\n1. Downloading/loading gene annotations...")
    if annotations_path is None:
        # Default to same directory as model_dir (typically "data" folder)
        data_dir = os.path.dirname(model_dir)
        os.makedirs(data_dir, exist_ok=True)
        annotations_path = os.path.join(data_dir, "scgpt_gene_info.csv")
    
    if not os.path.isfile(annotations_path):
        logger.info(f"   Downloading gene annotations from {SCGPT_DEFS.GENE_IDENTIFIERS_URL}")
        download_wget(SCGPT_DEFS.GENE_IDENTIFIERS_URL, annotations_path)
    
    gene_annotations = load_gene_annotations(annotations_path)
    logger.info(f"   Loaded {len(gene_annotations)} gene annotations")
    
    # 2. Load model
    logger.info("2. Loading scGPT model...")
    model, vocab, model_metadata, checkpoint_path = load_scgpt(model_dir)
    logger.info(f"   {len(gene_annotations)} genes, {model_metadata[FM_DEFS.N_LAYERS]} layers")

    # 3. Extract weights
    logger.info("3. Extracting weights...")
    weights = extract_model_weights(model, vocab, model_metadata, checkpoint_path)
    logger.info(f"   Embeddings: {weights.gene_embedding.shape}")
    logger.info(f"   Attention weights: {model_metadata[FM_DEFS.N_LAYERS]} layers × 4 matrices (Q,K,V,O)")

    # 4. Create FoundationModel and save
    logger.info("4. Creating FoundationModel and saving...")
    foundation_model = FoundationModel(
        weights=weights,
        gene_annotations=gene_annotations,
        model_metadata=model_metadata,
    )
    foundation_model.save(output_dir, file_prefix)
    logger.info("   Successfully saved all results!")

    return None

def load_scgpt(model_dir: str) -> Tuple[TransformerModel, Vocab, dict, str]:
    """
    Load the scGPT model from a directory containing model files.
    
    Parameters
    ----------
    model_dir : str
        Directory containing the scGPT model files (args.json, best_model.pt, vocab.json)
    
    Returns
    -------
    model : TransformerModel
        The loaded scGPT model
    vocab : Vocab
        The scGPT vocabulary
    model_metadata : dict
        Dictionary containing model metadata (model_name, n_genes, embed_dim, etc.)
    checkpoint_path : str
        Path to the checkpoint file (best_model.pt), useful for direct weight extraction
    """

    model_config_file = os.path.join(model_dir, SCGPT_DEFS.CONFIG_FILENAME)
    model_file = os.path.join(model_dir, SCGPT_DEFS.MODEL_FILENAME)
    vocab_file = os.path.join(model_dir, SCGPT_DEFS.VOCAB_FILENAME)

    vocab = GeneVocab.from_file(vocab_file)
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)

    # Retrieve model parameters from config files
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    print(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )

    model = load_scgpt_model(model_file, vocab, model_configs)

    model_metadata = format_model_metadata(model_configs, vocab)

    return model, vocab, model_metadata, model_file
    

def load_scgpt_model(model_file, vocab, model_configs) -> TransformerModel:

    """
    Load and return the scGPT model

    Parameters
    ----------
    model_file : str
        The file containing the scGPT model
    vocab : Vocab
        The scGPT vocabulary
    model_configs : dict
        The metadata for the scGPT model

    Returns
    -------
    model : TransformerModel
        The scGPT model
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        model_configs[SCGPT_DEFS.EMBSIZE],
        model_configs[SCGPT_DEFS.NHEAD],
        model_configs[SCGPT_DEFS.D_HID],
        model_configs[SCGPT_DEFS.NLAYERS],
        vocab=vocab,
        pad_value=PAD_VALUE,
        n_input_bins=N_INPUT_BINS,
    )

    try:
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        print(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location=torch.device('cpu'))
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    model.to(device)

    return model

def format_model_metadata(model_configs, vocab):

    """Summarize the model architecture and other metadata"""

    # Get vocabulary as list of tokens in order
    vocab_list = vocab.get_itos()
    
    # Count actual genes (excluding special tokens)
    n_genes = len([token for token in vocab_list if not token.startswith('<') and token != PAD_TOKEN])
    
    return {
        FM_DEFS.MODEL_NAME: SCGPT_DEFS.MODEL_NAME,
        FM_DEFS.N_GENES: n_genes,
        FM_DEFS.N_VOCAB: len(vocab),
        FM_DEFS.ORDERED_VOCABULARY: vocab_list,  # Vocabulary in order
        FM_DEFS.EMBED_DIM: model_configs[SCGPT_DEFS.D_HID],
        FM_DEFS.N_LAYERS: model_configs[SCGPT_DEFS.NLAYERS],
        FM_DEFS.N_HEADS: model_configs[SCGPT_DEFS.NHEAD]
    }

def load_gene_annotations(annotations_path) -> pd.DataFrame:

    """Load gene annotations"""

    return (
        pd.read_csv(annotations_path, index_col = 0)
        .rename(columns = {
            "feature_id" : ONTOLOGIES.ENSEMBL_GENE,
            "feature_name" : ONTOLOGIES.SYMBOL
        })
        .assign(**{FM_DEFS.VOCAB_NAME : lambda x: x[ONTOLOGIES.SYMBOL]})
        .drop(columns = ["feature_length", "soma_joinid"])
    )

def extract_model_weights(model, vocab, model_metadata, checkpoint_path) -> FoundationModelWeights:
    """
    Extract gene embeddings and attention weights from the scGPT model.
    
    Note: Weights must be loaded directly from the checkpoint file because
    model.state_dict() returns incorrect/shared weights across layers.
    
    The model IS loaded from the checkpoint (so keys match), but model.state_dict()
    returns identical weights for all layers. This suggests a weight sharing bug in
    scGPT's implementation where all layers reference the same Wqkv parameter object.
    When state_dict() is called, it returns the same parameter multiple times (once
    per layer key), but they all point to the same underlying tensor.
    
    Loading directly from the checkpoint file bypasses this bug and gives us the
    actual stored weights for each layer.
    
    Parameters
    ----------
    model : TransformerModel
        The scGPT model
    vocab : Vocab
        The scGPT vocabulary
    model_metadata : dict
        Model metadata dictionary containing N_LAYERS and EMBED_DIM
    checkpoint_path : str
        Path to checkpoint file. Required because model.state_dict() returns
        incorrect weights (all layers have identical values).
    
    Returns
    -------
    FoundationModelWeights
        FoundationModelWeights instance containing gene_embedding and attention_layers
    """
    # Extract gene embeddings
    gene_ids = torch.arange(len(vocab))
    embeddings = model.encoder(gene_ids).detach().cpu().numpy()
    
    # Load weights directly from checkpoint file (model.state_dict() is unreliable)
    # The checkpoint file contains the correct weights, but model.state_dict() 
    # returns identical values across all layers, suggesting a bug in scGPT's model
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract attention weights for all layers as AttentionLayer instances
    attention_layers = []
    n_layers = model_metadata[FM_DEFS.N_LAYERS]
    d = model_metadata[FM_DEFS.EMBED_DIM]
    
    for layer_idx in range(n_layers):
        # scGPT uses Wqkv.weight for the combined QKV projection
        wqkv_key = f'transformer_encoder.layers.{layer_idx}.self_attn.Wqkv.weight'
        out_proj_key = f'transformer_encoder.layers.{layer_idx}.self_attn.out_proj.weight'
        
        if wqkv_key not in state_dict:
            raise KeyError(f"Could not find {wqkv_key} in state_dict")
        if out_proj_key not in state_dict:
            raise KeyError(f"Could not find {out_proj_key} in state_dict")
        
        # Clone immediately to ensure independent copies
        in_proj = state_dict[wqkv_key].clone()
        out_proj = state_dict[out_proj_key].clone()
        
        # Validate shape
        if in_proj.shape[0] != 3 * d:
            raise ValueError(
                f"Expected in_proj.shape[0] to be 3*d ({3*d}), but got {in_proj.shape[0]}"
            )
        
        # Split QKV into separate matrices and convert to numpy
        w_q = in_proj[:d, :].clone().cpu().detach().numpy()
        w_k = in_proj[d:2*d, :].clone().cpu().detach().numpy()
        w_v = in_proj[2*d:, :].clone().cpu().detach().numpy()
        w_o = out_proj.clone().cpu().detach().numpy()
        
        attention_layers.append(
            AttentionLayer(
                layer_idx=layer_idx,
                W_q=w_q,
                W_k=w_k,
                W_v=w_v,
                W_o=w_o,
            )
        )
    
    return FoundationModelWeights(
        gene_embedding=embeddings,
        attention_layers=attention_layers,
    )
