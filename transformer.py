
import torch
from torch.nn import Embedding
from torch import nn
import lightning as L
import math
class AttentionHead(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.Q = nn.Linear(embedding_size,embedding_size)
        self.K = nn.Linear(embedding_size,embedding_size)
        self.V = nn.Linear(embedding_size,embedding_size)

    def forward(self, X):
        # E: embedding size
        # T: seq size 
        query = self.Q(X)
        key = self.K(X)
        value = self.V(X)
        # (T, E) (E, T)
        dims = X.ndim
        T_dim = dims-2
        E_dim = dims-1
        T = X.shape[T_dim]
        invert_key = torch.transpose(key, T_dim, E_dim)
        
        attn_scores = torch.matmul(query, invert_key)
        norm_scores = torch.div(attn_scores, math.sqrt(self.embedding_size))
        # for each token's attention score null out future tokens
        to_mask_indices = torch.triu_indices(T,T,offset=1)
        print(to_mask_indices)
        # TODO(Ian): Do something better here
        if dims > 2:
            norm_scores[:, to_mask_indices[0], to_mask_indices[1]] = -torch.inf
        else:
            norm_scores[to_mask_indices[0], to_mask_indices[1]] = -torch.inf 
        print(norm_scores)
        # (T,T)
        attn_matrix = torch.softmax(norm_scores, dim=dims-2)
        #T,T * T,E
        return torch.matmul(attn_matrix, value)


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, embedding_size):
        super().__init__()
        self.mods=[AttentionHead(embedding_size) for _ in range(num_heads)]

    def forward(self, X):
        # S, T, E
        dims = X.ndim
        # TODO(Ian): make this easily scalable by batching.
        return torch.cat([m(X) for m in self.mods], dims-1)

class FeedForwardNetwork(nn.Module):
    def __init__(self, input: int, hidden_layer_factor: int, output: int, bias: bool, dropout: float):
        super().__init__()
        self.hidden = nn.Linear(input, hidden_layer_factor* input, bias)
        self.gelu = nn.GELU()
        self.projections = nn.Linear(hidden_layer_factor * input, output, bias)
        self.drop = nn.Dropout(dropout)
    def forward(self, X):
        mid = self.hidden(X)
        proj = self.projections(self.gelu(mid))
        return self.drop(proj)

class Decoder(nn.Module):
    def __init__(self, num_heads, embedding_size, hidden_layer_factor: int,
                  bias: bool, dropout: float):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_head = num_heads
        self.output_size = self.num_head * self.embedding_size
        self.heads = MultiHeadedAttention(num_heads, embedding_size)
        self.pointwise_ff = FeedForwardNetwork(self.output_size, 
                                            hidden_layer_factor, 
                                            self.embedding_size,
                                            bias, dropout)
        self.lnorm1 = nn.LayerNorm(self.embedding_size)
        self.lnorm2 = nn.LayerNorm(self.output_size)

    def forward(self, X):
        attn = self.heads(self.lnorm1(X))
        activation = self.pointwise_ff(self.lnorm2(attn))
        return activation

class Model(nn.Module):
    def __init__(self,  vocab_size, _max_seq_len: int, num_decoders, num_heads, 
                 embedding_size, hidden_layer_factor: int,
                  bias: bool, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # TODO(Ian): do this
        #self.abs_position_embedding = nn.Embedding()
        self.decs = [Decoder(num_heads, embedding_size, hidden_layer_factor,
                             bias, dropout) for _ in range(num_decoders)]
        self.norm = nn.LayerNorm(embedding_size)
        self.classifier = nn.Linear(embedding_size, vocab_size, bias)

    def forward(self, X_idxs):
        embs = self.embedding(X_idxs)
        for dec in self.decs:
            embs = dec(embs)
        return self.classifier(self.norm(embs))


def main():
    athd = Model(500, 500, 2, 2, 256, 4, False, .1)
    mat = torch.randint(0, 100, (5,10))
    print(mat)
    print(mat.shape)
    X = athd.forward(mat)
    print(X)
    print(X.shape)
if __name__ == "__main__":
    main()