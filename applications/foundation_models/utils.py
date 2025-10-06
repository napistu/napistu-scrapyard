""" Common cross-model utility functions. """

from types import SimpleNamespace

import numpy as np
from scipy.special import softmax

RESULTS_DEFS = SimpleNamespace(
    N_GENES = "n_genes",
    EMBED_DIM = "embed_dim",
    N_LAYERS = "n_layers",
    N_HEADS = "n_heads",
)

def load_model_weights(path):
    data = np.load(path, allow_pickle=True)
    return {
        'genes': data['genes'].tolist(),
        'embeddings': data['embeddings'],
        'attention_weights': data['attention_weights'].item(),
        'metadata': data['metadata'].item()
    }

def compute_attention_from_weights(embeddings, W_q, W_k):
    """Same computation, just load weights from file"""
    Q = embeddings @ W_q.T
    K = embeddings @ W_k.T
    attn_scores = (Q @ K.T) / np.sqrt(Q.shape[-1])
    return softmax(attn_scores, axis=-1)
