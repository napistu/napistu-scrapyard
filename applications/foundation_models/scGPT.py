""" Utility functions for interacting with the scGPT model. """

from typing import Tuple
from types import SimpleNamespace

import json
import os
from pathlib import Path
import sys
import warnings

import torch
import numpy as np

from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

sys.path.insert(0, "../")
from scgpt.tasks import GeneEmbedding
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.utils import set_seed

from utils import RESULTS_DEFS

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

    EMBSIZE = model_configs[SCGPT_DEFS.EMBSIZE]
    NHEAD = model_configs[SCGPT_DEFS.NHEAD]
    D_HID = model_configs[SCGPT_DEFS.D_HID]
    NLAYERS = model_configs[SCGPT_DEFS.NLAYERS]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        EMBSIZE,
        NHEAD,
        D_HID,
        NLAYERS,
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

    return {
        RESULTS_DEFS.N_GENES: len(vocab),
        RESULTS_DEFS.EMBED_DIM: model_configs[SCGPT_DEFS.D_HID],
        RESULTS_DEFS.N_LAYERS: model_configs[SCGPT_DEFS.NLAYERS],
        RESULTS_DEFS.N_HEADS: model_configs[SCGPT_DEFS.NHEAD]
    }

def extract_model_weights(model, vocab, output_path):
    """Extract just the weights needed to compute attention"""
    
    # Gene embeddings
    gene_ids = torch.arange(len(vocab))
    embeddings = model.encoder(gene_ids).detach().cpu().numpy()
    
    # Attention weights for all layers
    attention_weights = {}
    for layer_idx in range(12):
        in_proj = model.transformer_encoder.layers[layer_idx].self_attn.in_proj_weight
        
        d = 512
        attention_weights[f'layer_{layer_idx}'] = {
            'W_q': in_proj[:d, :].cpu().detach().numpy(),
            'W_k': in_proj[d:2*d, :].cpu().detach().numpy(),
            'W_v': in_proj[2*d:, :].cpu().detach().numpy()
        }
    
    # Save everything
    data = {
        'model_name': 'scGPT',
        'genes': vocab.get_itos(),
        'embeddings': embeddings,
        'attention_weights': attention_weights,
        'metadata': {
            'n_genes': len(vocab),
            'embed_dim': 512,
            'n_layers': 12
        }
    }
    
    np.savez_compressed(output_path, **data)