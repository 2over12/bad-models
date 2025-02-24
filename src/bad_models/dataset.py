from datasets import load_dataset, Dataset
import lightning as L
from omegaconf import DictConfig
from pygtrie import PrefixSet 
import more_itertools
import json
import torch

UNKNOWN_TOKEN = "<UNK>"
STOP_TOKEN = "<STOP>"


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
    def __init__(self, tokens: list[str]):
        self.trie = PrefixSet(tokens)
        self.tokens = sorted(tokens)
        self.token_to_ind = dict([(tok, i) for i, tok in enumerate(self.tokens)])


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
            if curr_tok not in self.trie:
                is_UNK = True
                while is_UNK and it.peek(default=None) is not None:
                    n = it.peek()
                    is_UNK = n not in self.trie
                    if is_UNK:
                        next(it)
                total.append(UNKNOWN_TOKEN)
                curr_tok = ""
                continue

            nxt = it.peek(default=None)
            if nxt is None or (curr_tok + nxt) not in self.trie:
                total.append(curr_tok)
                curr_tok = ""
        return total

    def tokenize(self, input: str) -> list[str]:
        normed = normalize(input)
        wds = words(normed)
        return [tok for wd in wds for tok in self.get_tokens_for_wd(wd)]

    def token_index(self, tok: str) -> int | None:
        return self.token_to_ind.get(tok, None)

class OrcaModule(L.LightningDataModule):
    def __init__(self, batch_size: int, toks: Tokenizer):
        super().__init__()
        self.batch_size = batch_size
        self.toks = toks

    def prepare_data(self):
        def tokenize(row):
            q = row["question"]
            return {"question": [self.toks.token_index(x) for x in self.toks.tokenize(q)]}
        ds = load_dataset(DATASET_NAME, split="train").map(tokenize)


    def train_dataloader(self):
        return super().train_dataloader()

def main():
    tok = Tokenizer([UNKNOWN_TOKEN, STOP_TOKEN, "blah", "x", "foo"])
    mod = OrcaModule(4, tok)
    mod.prepare_data()

if __name__ == "__main__":
    main()