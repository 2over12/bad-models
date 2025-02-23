
import torch
from torch.nn import Embedding
from torch import nn
import lightning as L
import math
from torch.utils.data import DataLoader
from tokenizer import Tokenizer, STOP_TOKEN
import copy

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.Q = nn.Linear(embedding_size,embedding_size)
        self.K = nn.Linear(embedding_size,embedding_size)
        self.V = nn.Linear(embedding_size,embedding_size)
        self.H = num_heads

    def split_heads(self, X: torch.Tensor):
        # S,T,E-> S,H,T E//H
        return X.view((X.shape[0], X.shape[1], self.H, X.shape[2]//self.H)).transpose(1,2)

    def forward(self, X):
        # E: embedding size
        # T: seq size 
        query = self.split_heads(self.Q(X))
        key =  self.split_heads(self.K(X))
        value =  self.split_heads(self.V(X))
        # (S,H,T,E//H) (S,H,E//H, T)
        dims = query.ndim
        T_dim = dims-2
        E_h_dim = dims-1
        T = query.shape[T_dim]
        invert_key = torch.transpose(key, T_dim, E_h_dim)
        
        # We want (S, H, T, E//H)
        # (S, H, E//H, T)
        attn_scores = torch.matmul(query, invert_key)
        norm_scores = torch.div(attn_scores, math.sqrt(self.embedding_size))
        # for each token's attention score null out future tokens
        to_mask_indices = torch.triu_indices(T,T,offset=1)
        #print(to_mask_indices)
        norm_scores[:,:, to_mask_indices[0], to_mask_indices[1]] = -torch.inf

        # (S, H, T,T)
        attn_matrix = torch.softmax(norm_scores, dim=dims-2)
        # (S, H, T, T) * (S, H, T, E//T) - >(S, H, T, E//T)
        values = torch.matmul(attn_matrix, value)
        # flip the heads so that each head is next to it
        cont = values.transpose(1,2)#.contiguous()
        concats = cont.reshape(X.shape[0], T, self.embedding_size)
        return concats

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
        self.output_size = self.embedding_size
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
    def __init__(self,  vocab_size, num_decoders, num_heads, 
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
        classed = self.classifier(self.norm(embs))
        maxed = torch.nn.functional.softmax(classed, 2)
        return maxed


class GRPOTraining(L.LightningModule):
    def __init__(self, vocab_size, num_decoders, num_heads, 
                 embedding_size, hidden_layer_factor: int,
                  bias: bool, G: int, tokenizer: Tokenizer,
                  grpo_steps=10,
                  max_tok_sq_len=None, dropout=0.0, epsilon=.2, kl_div_weight=.5):
        super().__init__()
        self.automatic_optimization = False
        # TODO(Ian): investigate results on this... i kinda assume 
        # dropout is not ideal here since it will really mess with 
        # RL exploration as probabilies fluxate but maybe if the model
        # stabilizes a bit in training maybe it's ideal? somebody probably knows
        self.mod = Model(vocab_size, num_decoders, num_heads, 
                 embedding_size, hidden_layer_factor,
                  bias, dropout)
        self.copy_mod = copy.deepcopy(self.mod)
        self.copy_mod.requires_grad_(False)
        self.G = G
        self.tokenizer = tokenizer
        self.max_tok_sq_len = max_tok_sq_len
        self.grpo_steps = grpo_steps
        self.epsilon = epsilon
        self.kl_div_weight = kl_div_weight
    
    # TODO(Ian): make this abstract
    def score_input(self, row: torch.Tensor) -> bool:
        return torch.randint(0, 10, (1,), dtype=torch.float)

    def all_stopped(self, X: torch.Tensor) -> bool:
        if self.max_tok_sq_len is not None and X.shape[1] >= self.max_tok_sq_len:
                return True

        stop_ind = self.tokenizer.token_index(STOP_TOKEN)
        tany = torch.any(torch.eq(X, stop_ind), 0)
        return not torch.all(torch.logical_not(tany))
    
    def predictions(self, X: torch.Tensor):
        preds = self.mod(X)[:, -1, :]
        selected = torch.multinomial(preds, 1)
        return selected
   
    def seq_to_probs(self, X: torch.Tensor, start_len: int):
        # we have Q_0 ... Q_N, T_0 ... T_L
        # where start_len == N + 1 
        # so we want to subtract 1 to get the first token prob we consider as 
        # The next token T_0
        # we also do -1 to drop T_L since we dont have a next token
        preds = self.mod(X)[:,start_len-1:-1,:]
        # now we select the next token index for each token
        toks = X[:,start_len:]
        # now select the probability of each next toekn from our distribution
        toksv = toks.view((toks.shape[0], toks.shape[1], 1))
        # TODO(Ian): this doesnt explicitly happen in their objective function but
        # calculuting the probability in log form  the exp(-) to grab the ratio
        probs = torch.gather(preds, 2, toksv)
        return torch.sum(torch.log(probs.view(probs.shape[0], probs.shape[1])), 1)
   

    def kl_divergence_penalty(self, state_curr, state_prev): 
        return torch.mean(state_prev * (torch.log(state_prev) - torch.log(state_curr)))

    # TODO(Ian): padding mask here
    # TODO(Ian): batching
    # Note this function as written can only take a single prompt at shape
    # (1, T)
    def training_step(self, batch: torch.Tensor, _idx, training=True):
        start_len = batch.shape[1]
        repeated = batch.repeat(self.G, 1)
        while not self.all_stopped(repeated):
            sels = self.predictions(repeated)
            repeated = torch.cat((repeated, sels), 1)
        scores = torch.func.vmap(self.score_input, 0, 0, randomness="different")(repeated).view(repeated.shape[0])
        advantage = (scores - torch.mean(scores)) / torch.std(scores)
        old_probs = self.seq_to_probs(repeated, start_len).detach()
        state_res = self.copy_mod(repeated)
        opt = None
        if training:
            opt = self.optimizers()
        for i in range(self.grpo_steps):
            self.zero_grad()
            state_curr = self.mod(repeated)
            new_probs = self.seq_to_probs(repeated, start_len)
            probs = torch.exp(new_probs - old_probs)
            flat_advantage = probs * advantage
            clipped_prob = torch.clip(probs, 1 - self.epsilon, 1 + self.epsilon)
            clipped_adantage = clipped_prob * advantage
            # TODO(Ian): double check the background here. it seems the assumption
            # is to take a heavily negative value in order to heavily reject
            # poor advantages and bias towards avoiding bad actions
            actual_advantage = torch.minimum(flat_advantage,clipped_adantage)
            clipped_loss = torch.mean(actual_advantage)
    
            if training:
                self.manual_backward(-clipped_loss + self.kl_div_weight * self.kl_divergence_penalty(state_curr, state_res),retain_graph=True)
                opt.step()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

class Trainable(L.LightningModule):
    def __init__(self, vocab_size, num_decoders, num_heads, 
                 embedding_size, hidden_layer_factor: int,
                  bias: bool, dropout: float):
        super().__init__()
        self.mod = Model(vocab_size, num_decoders, num_heads, 
                 embedding_size, hidden_layer_factor,
                  bias, dropout)

    # TODO(Ian): padding mask here
    def training_step(self, batch, _idx):
        x, y = batch
        res = self.mod(x)

        xres = res.view(res.shape[0]*res.shape[1], res.shape[2])
        yres = y.view((res.shape[0]*res.shape[1],))
        return torch.nn.functional.cross_entropy(xres, yres)


    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer



def main():
    # S = 1
    # T = 3
    # E = 10
    # mat = torch.rand((S, T, E))
    # athd = MultiHeadedAttention(2,E)
    # re_embedded = athd(mat)
    # print(re_embedded.shape)
    athd = GRPOTraining(500, 2, 2, 256, 4, False, 5, Tokenizer([STOP_TOKEN]),
                       max_tok_sq_len=20)
    mat = torch.randint(0, 100, (1,5))
    #res = torch.randint(0,100, (2, 5))
    #athd.training_step(mat, 1)
    ldr = DataLoader(mat)
    #other = DataLoader(res)
    trainer = L.Trainer(detect_anomaly=False)
    trainer.fit(athd, train_dataloaders=ldr)
if __name__ == "__main__":
    main()