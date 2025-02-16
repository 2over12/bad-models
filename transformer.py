
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
        return torch.cat([m(X) for m in self.mods], dims-1)

def main():
    athd = MultiHeadedAttention(2, 256)
    mat = torch.rand((5, 256), requires_grad=True)
    print(mat)
    print(mat.shape)
    X = athd.forward(mat)
    print(X)
    print(X.shape)
if __name__ == "__main__":
    main()