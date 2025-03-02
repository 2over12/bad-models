
import torch
from torch.nn import Embedding
from torch import nn
import lightning as L
import math
from torch.utils.data import DataLoader
from bad_models.dataset import Tokenizer, STOP_TOKEN, OrcaModule
import copy
import hydra
from omegaconf import DictConfig

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, embedding_size, lora_rank=None):
        super().__init__()
        self.embedding_size = embedding_size
        self.Q = nn.Linear(embedding_size,embedding_size)
        self.K = nn.Linear(embedding_size,embedding_size)
        self.V = nn.Linear(embedding_size,embedding_size)
        self.LQ = LoraLayer(embedding_size, embedding_size, lora_rank)
        self.LK = LoraLayer(embedding_size, embedding_size, lora_rank)
        self.LV = LoraLayer(embedding_size, embedding_size, lora_rank)
        if lora_rank is not None:
            self.Q.requires_grad_(False)
            self.K.requires_grad_(False)
            self.V.requires_grad_(False)

        self.H = num_heads

    def split_heads(self, X: torch.Tensor):
        # S,T,E-> S,H,T E//H
        return X.view((X.shape[0], X.shape[1], self.H, X.shape[2]//self.H)).transpose(1,2)

    
    def apply_kqv(self, X, linear: nn.Linear, lora: "LoraLayer"):
        prev_res = linear(X)
        return lora(X, prev_res)

    def forward(self, X, mask: torch.Tensor | None =None):
        # mask: (S, H, T)
        # E: embedding size
        # T: seq size 
        query = self.split_heads(self.apply_kqv(X, self.Q, self.LQ))
        key =  self.split_heads(self.apply_kqv(X, self.K, self.LK))
        value =  self.split_heads(self.apply_kqv(X, self.V, self.LV))
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

        # (S, H, T, T)
        attn_matrix = torch.softmax(norm_scores, dim=dims-2)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, self.H, T, 1)

        masked_attn_matrix = attn_matrix * mask if mask is not None else attn_matrix
 
        # (S, H, T, T) * (S, H, T, E//T) - >(S, H, T, E//T)
        values = torch.matmul(masked_attn_matrix, value)
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


class LoraLayer(nn.Module):
    def __init__(self, input_size, output_size, rank):
        super().__init__()
        self.A = None
        self.B = None
        if rank is not None:
            self.A = nn.Linear(input_size, rank, bias=False)
            self.B = nn.Linear(rank, output_size, bias=False)
        

    def forward(self, X, prev_res):
        if self.A is None or self.B is None:
            return prev_res
        return self.B(self.A(X)) + prev_res


class Decoder(nn.Module):
    def __init__(self, num_heads, embedding_size, hidden_layer_factor: int,
                  bias: bool, dropout: float, lora_rank=None):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_head = num_heads
        self.output_size = self.embedding_size
        self.heads = MultiHeadedAttention(num_heads, embedding_size, lora_rank=lora_rank)
        self.pointwise_ff = FeedForwardNetwork(self.output_size, 
                                            hidden_layer_factor, 
                                            self.embedding_size,
                                            bias, dropout)
        
        if lora_rank is not None:
            self.pointwise_ff.requires_grad_(False)

        self.lora_layer_ff = LoraLayer(self.output_size, self.embedding_size, lora_rank)

        self.lnorm1 = nn.LayerNorm(self.embedding_size)
        self.lnorm2 = nn.LayerNorm(self.output_size)

    def forward(self, X, mask=None):
        attn = self.heads(self.lnorm1(X), mask=mask)
        x = self.lnorm2(attn)
        activation = self.pointwise_ff(x)
        return self.lora_layer_ff(x, activation)

class Model(nn.Module):
    def __init__(self,  vocab_size, num_decoders, num_heads, 
                 embedding_size, hidden_layer_factor: int,
                  bias: bool, dropout: float, lora_rank=None):
        super().__init__()
        self.embedding_size  = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.decs = [Decoder(num_heads, embedding_size, hidden_layer_factor,
                             bias, dropout, lora_rank=lora_rank) for _ in range(num_decoders)]
        self.norm = nn.LayerNorm(embedding_size)
        self.classifier = nn.Linear(embedding_size, vocab_size, bias)

    def pos_embeddings(self, positions: torch.Tensor):
        # (B, T) we allow arbitrary positions in order to allow shifting pads etc
        shp = positions.shape
        embs = torch.zeros((shp[0] * shp[1], self.embedding_size))
        
        # only need half since 2k, 2k+1 collapse
        inds = torch.arange(0, self.embedding_size//2)
        denom = torch.pow(10_000, inds*2/self.embedding_size)
        
        # (B,T) (E//2)
        posreps = positions.unsqueeze(-1).expand(-1, -1, self.embedding_size//2)
        
        evens = torch.sin(posreps.flatten(end_dim=1)/denom)
        odds = torch.cos(posreps.flatten(end_dim=1)/denom)
        embs[:, 0::2] = evens
        embs[:, 1::2] = odds
        return embs.view((shp[0], shp[1], self.embedding_size))


    def forward(self, X_idxs_and_pos_mask):
        X_idxs, pos, mask = X_idxs_and_pos_mask
        embs = self.embedding(X_idxs) + self.pos_embeddings(pos)
        for dec in self.decs:
            embs = dec(embs, mask=mask)
        classed = self.classifier(self.norm(embs))
        maxed = torch.nn.functional.softmax(classed, 2)
        return maxed


class GRPOTraining(L.LightningModule):
    def __init__(self, vocab_size, num_decoders, num_heads, 
                 embedding_size, hidden_layer_factor: int,
                  bias: bool, G: int, tokenizer: Tokenizer,
                  grpo_steps=10, lora_rank = None,
                  max_tok_sq_len=None, dropout=0.0, epsilon=.2, kl_div_weight=.5):
        super().__init__()
        self.automatic_optimization = False
        # TODO(Ian): investigate results on this... i kinda assume 
        # dropout is not ideal here since it will really mess with 
        # RL exploration as probabilies fluxate but maybe if the model
        # stabilizes a bit in training maybe it's ideal? somebody probably knows
        self.mod = Model(vocab_size, num_decoders, num_heads, 
                 embedding_size, hidden_layer_factor,
                  bias, dropout, lora_rank=lora_rank)
        self.copy_mod = copy.deepcopy(self.mod)
        self.copy_mod.requires_grad_(False)
        self.G = G
        self.tokenizer = tokenizer
        self.max_tok_sq_len = max_tok_sq_len
        self.grpo_steps = grpo_steps
        self.epsilon = epsilon
        self.kl_div_weight = kl_div_weight
    
    # TODO(Ian): make this abstract
    # This is just like stubbedish format requirement
    def score_input(self, row: torch.Tensor) -> bool:
        lst = row.clone().detach().tolist()
        built_string: str = " ".join(self.tokenizer.decode(lst))
        start = built_string.startswith("<thinking>")
        end = "</thinking>" in built_string
        return torch.tensor([5.0 if start and end else 0.0])

    def all_stopped(self, X: torch.Tensor) -> bool:
        if self.max_tok_sq_len is not None and X.shape[2] >= self.max_tok_sq_len:
                return True

        stop_ind = self.tokenizer.token_index(STOP_TOKEN)
        tany = torch.any(torch.eq(X.flatten(end_dim=1), stop_ind), 1)
        return torch.all(tany)
    

    def batch_tensor(self, X: torch.Tensor):
        batched = X.view(X.shape[0] * X.shape[1], X.shape[2])
        return batched


    def predictions(self, X: torch.Tensor, pos, mask):
        batched = self.batch_tensor(X)
        # batch preds
        # (G*B, T)
        preds = self.mod((batched, pos, mask))[:, -1, :]
        selected = torch.multinomial(preds, 1)
        shape_sel = selected.view(X.shape[0], X.shape[1], 1)
        return shape_sel
   

    def seq_to_probs(self, X: torch.Tensor, start_len: int, pos, mask):
        #X : (B, G, T)
        batched = self.batch_tensor(X)
        # we have Q_0 ... Q_N, T_0 ... T_L
        # where start_len == N + 1 
        # so we want to subtract 1 to get the first token prob we consider as 
        # The next token T_0
        # we also do -1 to drop T_L since we dont have a next token
        preds = self.mod((batched, pos, mask))[:,start_len-1:-1,:]
        # now we select the next token index for each token
        toks = batched[:,start_len:]
        # now select the probability of each next toekn from our distribution
        toksv = toks.view((toks.shape[0], toks.shape[1], 1))
        # TODO(Ian): this doesnt explicitly happen in their objective function but
        # calculuting the probability in log form  the exp(-) to grab the ratio
        probs = torch.gather(preds, 2, toksv)
        aggregated =  torch.sum(torch.log(probs.view(probs.shape[0], probs.shape[1])), 1)
        return aggregated.view((X.shape[0], X.shape[1]))

    def kl_divergence_penalty(self, state_curr, state_prev): 
        return torch.mean(state_prev * (torch.log(state_prev) - torch.log(state_curr)))

    
    
    def score_batch(self, batch: torch.Tensor):
        tot = []
        for row in batch:
            tot.append(self.score_input(row))
        
        return torch.cat(tot)

    # TODO(Ian): padding mask here
    # TODO(Ian): batching
    # Note this function as written can only take a single prompt at shape
    # (1, T)
    def training_step(self, batch_pos: torch.Tensor, _idx, training=True):
        batch, pos, mask = batch_pos
        # pos: (B, T)
        start_len = batch.shape[1]
        # batch: (B, T) -> (B, 1, T) -> (B, G, T)
        repeated = batch.view(batch.shape[0], 1, start_len).repeat(1, self.G, 1)
        # (B*G, T)
        repeated_pos = pos.view((pos.shape[0], 1 , pos.shape[1])).repeat(1, self.G, 1).flatten(end_dim=1)
        # (B*G, T)
        repeated_mask = mask.view((mask.shape[0], 1 , mask.shape[1])).repeat(1, self.G, 1).flatten(end_dim=1)

        while not self.all_stopped(repeated):
            sels = self.predictions(repeated, repeated_pos, repeated_mask)
            repeated = torch.cat((repeated, sels), 2)
            inc_last = repeated_pos[:,-1] + 1
            repeated_pos = torch.cat((repeated_pos, inc_last.unsqueeze(-1)),1)
            added = torch.ones((repeated_mask.shape[0], 1))
            repeated_mask = torch.cat((repeated_mask, added), 1)

        scores = self.score_batch(self.batch_tensor(repeated)).view(repeated.shape[0], repeated.shape[1])
        advantage = (scores - torch.mean(scores)) / torch.std(scores)
        old_probs = self.seq_to_probs(repeated, start_len, repeated_pos, repeated_mask).detach()
        state_res = self.copy_mod((self.batch_tensor(repeated), repeated_pos, repeated_mask))
        opt = None
        if training:
            opt = self.optimizers()
        for _ in range(self.grpo_steps):
            self.zero_grad()
            state_curr = self.mod((self.batch_tensor(repeated), repeated_pos, repeated_mask))
            new_probs = self.seq_to_probs(repeated, start_len, repeated_pos, repeated_mask)
            probs = torch.exp(new_probs - old_probs)
            flat_advantage = probs * advantage
            clipped_prob = torch.clip(probs, 1 - self.epsilon, 1 + self.epsilon)
            clipped_adantage = clipped_prob * advantage
            # TODO(Ian): double check the background here. it seems the assumption
            # is to take a heavily negative value in order to heavily reject
            # poor advantages and bias towards avoiding bad actions
            actual_advantage = torch.minimum(flat_advantage,clipped_adantage)
            # TODO(Ian): is mean across groups fine here, seems ok to me tbh?
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
        seqs, pos, mask = batch
        # Seqs: B *  L
        # the next tokens are shifted forward 1
        y = seqs[:,1:]

        # we trim the last token since there is no nex token
        res = self.mod((seqs, pos, mask))
        # res: B * L * C

        # mask based on the future reshaped to make multipication work
        rep_mask = mask[:,1:].unsqueeze(2).repeat(1,1,res.shape[2])
        # mask the gradients on predictions that should predict pad
        trimres = res[:, :-1, :]*rep_mask
        xres = trimres.reshape(trimres.shape[0]*trimres.shape[1], trimres.shape[2])
        yres = y.reshape((trimres.shape[0]*trimres.shape[1],))
        return torch.nn.functional.cross_entropy(xres, yres)


    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

class ModelBuilder:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def build(self) -> Trainable:
        return Trainable(self.cfg["tokenizer"]["vocab_size"], 
                     self.cfg["transformer"]["num_decoders"],
                     self.cfg["transformer"]["num_heads"],
                     self.cfg["embeddings"]["size"],
                     self.cfg["transformer"]["hidden_layer_factor"],
                     self.cfg["transformer"]["bias"],
                     self.cfg["transformer"]["dropout"])
    
class GRPOBuilder:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def build(self, toks: Tokenizer) -> GRPOTraining:
        return GRPOTraining(self.cfg["tokenizer"]["vocab_size"], 
                     self.cfg["transformer"]["num_decoders"],
                     self.cfg["transformer"]["num_heads"],
                     self.cfg["embeddings"]["size"],
                     self.cfg["transformer"]["hidden_layer_factor"],
                     self.cfg["transformer"]["bias"],
                     self.cfg["transformer"]["group_size"], toks,
                     max_tok_sq_len=self.cfg["transformer"]["max_grpo_seq_len"],
                     lora_rank=self.cfg["transformer"].get("lora_rank", None))

@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def train_base_model(cfg: DictConfig):
    mod = ModelBuilder(cfg).build()
    trainer = L.Trainer(detect_anomaly=False)
    trainer.fit(mod, train_dataloaders=OrcaModule(10, Tokenizer.from_file("tokens.txt")))


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def train_grpo(cfg: DictConfig):
    toks = Tokenizer.from_file("tokens.txt")
    mod = GRPOBuilder(cfg).build(toks)
    trainer = L.Trainer(detect_anomaly=False)
    trainer.fit(mod, train_dataloaders=OrcaModule(10, toks))


# def main():
#     # S = 1
#     # T = 3
#     E = 10
#     # mat = torch.rand((S, T, E))
#     #athd = MultiHeadedAttention(2,E)
#     # re_embedded = athd(mat)
#     # print(re_embedded.shape)
#     athd = GRPOTraining(500, 2, 2, 256, 4, False, 5, Tokenizer([STOP_TOKEN]),
#                        max_tok_sq_len=20)
    
#     tmodel = Model(500, 2, 2, 256, 4, True, 0.0)
#     mat = torch.randint(0, 100, (10,3))
#     pos = torch.randint(0, 5, (10, 3))
#     masks = torch.arange(0, 3).unsqueeze(0).repeat(10, 1)
#     #res = torch.randint(0,100, (2, 5))
#     #athd.training_step((mat, pos), 1, training=False)
#     ldr = DataLoader(mat)
#     pos_ldr = DataLoader(pos)
#     #other = DataLoader(res)
#     trainer = L.Trainer(detect_anomaly=False)
#     trainer.fit(athd, train_dataloaders=(ldr, pos_ldr, DataLoader(masks)))
# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    train_grpo()