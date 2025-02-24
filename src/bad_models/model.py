import os
from torch.nn import Embedding
from torch import nn
import lightning as L

class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings, size):
        self.emb = Embedding(num_embeddings=num_embeddings, embedding_dim=size)