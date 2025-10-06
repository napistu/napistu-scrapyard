""" Utility functions for interacting with the scGPT model. """

from typing import Tuple
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

# local .py file
from utils import (
    ONTOLOGIES,
    RESULTS_DEFS
)

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
    MODEL_NAME = "scGPT",
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

def load_scgpt(model_dir: str) -> Tuple[TransformerModel, Vocab, dict]:

    """
    Load the scGPT model and return the model and vocab

    Parameters
    ----------
    model_dir : str
        The directory containing the scGPT model

    Returns
    -------
    model : TransformerModel
        The scGPT model
    vocab : Vocab
        The scGPT vocabulary
    model_metadata : dict
        The metadata for the scGPT model
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

    return model, vocab, model_metadata
    

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
        RESULTS_DEFS.MODEL_NAME: SCGPT_DEFS.MODEL_NAME,
        RESULTS_DEFS.N_GENES: n_genes,
        RESULTS_DEFS.N_VOCAB: len(vocab),
        RESULTS_DEFS.ORDERED_VOCABULARY: vocab_list,  # Vocabulary in order
        RESULTS_DEFS.EMBED_DIM: model_configs[SCGPT_DEFS.D_HID],
        RESULTS_DEFS.N_LAYERS: model_configs[SCGPT_DEFS.NLAYERS],
        RESULTS_DEFS.N_HEADS: model_configs[SCGPT_DEFS.NHEAD]
    }

def load_gene_annotations(annotations_path) -> pd.DataFrame:

    """Load gene annotations"""

    return (
        pd.read_csv(annotations_path, index_col = 0)
        .rename(columns = {
            "soma_joinid" : RESULTS_DEFS.GENE_INDEX,
            "feature_id" : ONTOLOGIES.ENSEMBL_GENE,
            "feature_name" : ONTOLOGIES.SYMBOL
        })
        .drop(columns = "feature_length")
    )

def extract_model_weights(model, vocab, model_metadata):
    """
    Extract just the weights needed to compute attention
    
    Parameters
    ----------
    model : TransformerModel
        The scGPT model
    vocab : Vocab
        The scGPT vocabulary
    
    Returns
    -------
    dict
        A dictionary containing the gene embeddings and attention weights
    """
    
    # Gene embeddings
    gene_ids = torch.arange(len(vocab))
    embeddings = model.encoder(gene_ids).detach().cpu().numpy()
    
    # Attention weights for all layers
    attention_weights = {}
    for layer_idx in range(12):
        layer = model.transformer_encoder.layers[layer_idx]
        in_proj = layer.self_attn.in_proj_weight
        out_proj = layer.self_attn.out_proj.weight
        
        d = model_metadata[RESULTS_DEFS.EMBED_DIM]
        if in_proj.shape[0] != 3*d:
            raise ValueError(f"Expected in_proj.shape[0] to be 3*d, but got {in_proj.shape[0]}")

        attention_weights[RESULTS_DEFS.LAYER_NAME_TEMPLATE.format(layer_idx=layer_idx)] = {
            RESULTS_DEFS.W_Q: in_proj[:d, :].cpu().detach().numpy(),
            RESULTS_DEFS.W_K: in_proj[d:2*d, :].cpu().detach().numpy(),
            RESULTS_DEFS.W_V: in_proj[2*d:, :].cpu().detach().numpy(),
            RESULTS_DEFS.W_O: out_proj.cpu().detach().numpy()
        }
    
    return {
        RESULTS_DEFS.GENE_EMBEDDING : embeddings,
        RESULTS_DEFS.ATTENTION_WEIGHTS : attention_weights,
    }
