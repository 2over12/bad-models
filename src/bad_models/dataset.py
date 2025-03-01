from datasets import load_dataset, Dataset
import lightning as L
from omegaconf import DictConfig
from pygtrie import PrefixSet 
import more_itertools
import json
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import functools
import torch
UNKNOWN_TOKEN = "<UNK>"
STOP_TOKEN = "<STOP>"
PAD_TOKEN = "<PAD>"


def normalize(input: str):
    noleads = input.strip()
    return " ".join(noleads.split())

def words(input:str) -> list[str]:
    return input.split()


DATASET_NAME = "microsoft/orca-math-word-problems-200k"
def get_train_dataset(slc:int) -> Dataset:
    ds = load_dataset(DATASET_NAME, split="train")
    
    return ds[0: slc]



class Tokenizer:
    @staticmethod
    def from_file(pth: str) -> "Tokenizer":
        with open(pth, "r") as f:
            return Tokenizer(f.readlines() + [UNKNOWN_TOKEN, STOP_TOKEN, PAD_TOKEN])
    
    def __init__(self, tokens: list[str]):
        self.trie = PrefixSet(tokens)
        self.tokens = sorted(tokens)
        self.token_to_ind = dict([(tok, i) for i, tok in enumerate(self.tokens)])\

    def set_contains(self, wd: str) -> bool:
        res = self.trie._trie.longest_prefix(wd)
        res is not None and res == wd

    @staticmethod
    def from_dict(cfg: DictConfig):
        with open(cfg.tokenizer.dict, "r") as f:
            return Tokenizer(json.load(f))

    def get_tokens_for_wd(self, wd: str) -> list[str]:
        total = []
        curr_tok = ""
        it = more_itertools.peekable(wd)
        while it.peek(default=None) is not None:
            tok = next(it)
            curr_tok += tok
            # in this case we restarted and are forced to UNK
            if not self.set_contains(curr_tok):
                is_UNK = True
                while is_UNK and it.peek(default=None) is not None:
                    n = it.peek()
                    is_UNK = not self.set_contains(n)
                    if is_UNK:
                        next(it)
                total.append(UNKNOWN_TOKEN)
                curr_tok = ""
            else:
                nxt = it.peek(default=None)
                if nxt is None or not self.set_contains((curr_tok + nxt)):
                    assert curr_tok in self.trie
                    total.append(curr_tok)
                    curr_tok = ""
        return total

    def decode(self, toks: list[int]) -> list[str]:
        return [self.tokens[x] for x in toks]

    def tokenize(self, input: str, add_stop=True) -> list[str]:
        normed = normalize(input)
        stoken = [STOP_TOKEN] if add_stop else []
        wds = words(normed)
        return [tok for wd in wds for tok in self.get_tokens_for_wd(wd)] + stoken

    def token_index(self, tok: str) -> int | None:
        res = self.token_to_ind.get(tok, None)
        return res


def tokenize(toks, row):
    q = row["question"]
    return {"question": [toks.token_index(x) for x in toks.tokenize(q)]}

class OrcaModule(L.LightningDataModule):
    def __init__(self, batch_size: int, toks: Tokenizer):
        super().__init__()
        self.batch_size = batch_size
        self.toks = toks


    def load_ds(self):
        tok = self.toks
        f = functools.partial(tokenize, tok)
        return load_dataset(DATASET_NAME, split="train").map(f)

    def prepare_data(self):
        self.load_ds()

    def collate(self, batch):
        tens = [torch.tensor(x) for x in batch]
        seq = pad_sequence(tens, batch_first=True, padding_value=self.toks.token_index(PAD_TOKEN))

        def produce_mask(tens: torch.Tensor) -> torch.Tensor:
            pd_value = self.toks.token_index(PAD_TOKEN)
            cpy = tens.clone()
            cpy[cpy==pd_value] = 0.0
            cpy[cpy!=pd_value] = 1.0
            return cpy

        mask = produce_mask(seq)
        pos = torch.arange(0, mask.shape[1]).unsqueeze(0).repeat(mask.shape[0],1)
        return (seq, pos, mask)

    def train_dataloader(self):
        ds = self.load_ds()
        return DataLoader(ds["question"], batch_size=self.batch_size, collate_fn=self.collate)

def main():
    tok = Tokenizer([UNKNOWN_TOKEN, STOP_TOKEN, "blah", "x", "foo"])
    mod = OrcaModule(4, tok)
    mod.prepare_data()
    mod.train_dataloader()

if __name__ == "__main__":
    main()