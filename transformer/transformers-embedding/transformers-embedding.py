import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    weight = torch.randn(vocab_size, d_model)
    nn.init.xavier_uniform_(weight)
    return nn.Embedding.from_pretrained(weight)

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    d_model = torch.tensor(d_model)
    return embedding(tokens) * torch.sqrt(d_model)