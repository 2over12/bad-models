import pandas as pd
import more_itertools
import dataset
from omegaconf import DictConfig
import hydra
from dataclasses import dataclass
import json
from pygtrie import PrefixSet

UNKNOWN_TOKEN = "<UNK>"

def normalize(input: str):
    noleads = input.strip()
    return " ".join(noleads.split())

def words(input:str) -> list[str]:
    return input.split()


def compute_word_freqs(words: list[str]) -> dict[str, int]:
    freqs = {}

    for word in words:
        freqs[word] = freqs.get(word, 0) + 1
    return freqs


def base_tokens(words: list[str]) -> set[str]:
    toks = set()
    for word in words:
        for ch in word:
            toks.add(ch)
    return toks


def freq_of_token_pairs(split_words: dict[str, list[str]], word_freqs: dict[str, int]) -> dict[(str, str), int]:
    freqs = {}
    for wd, tokens in split_words.items():
        for (fst, snd) in more_itertools.windowed(tokens, 2):
            if fst is not None and snd is not None:
                it = (fst,snd)
                freqs[it] = freqs.get(it, 0) + word_freqs[wd]
    return freqs

def merge_adjacent(pr: tuple[str,str], split_words: dict[str, list[str]]):
    for wd, tokens in split_words.items():
        # Greedy
        complete_word = []
        lookahead = None
        for tok in tokens:
            if lookahead is not None and pr[1] == tok:
                complete_word += [str(pr[0] + pr[1])]
                lookahead = None
                continue

            if lookahead is not None:
                complete_word += [lookahead]
                lookahead = None
            
            if tok == pr[0]:
                lookahead = tok
            else:
                complete_word += [tok]
        
        if lookahead is not None:
            complete_word += [lookahead]
        
        split_words[wd] = complete_word

def bpe_train(words: list[str], target_vocab_size: int) -> set[str]:
    vocab = base_tokens(words) 
    vocab.add(UNKNOWN_TOKEN)
    split_words = {wd: [ch for ch in wd] for wd in words}
    word_freqs = compute_word_freqs(words)
    while len(vocab) < target_vocab_size:
        pair_freqs = freq_of_token_pairs(split_words, word_freqs)
        
        if len(pair_freqs) == 0:
            break
        
        (pr, _) = max(pair_freqs.items(), key=lambda tup: tup[1])
        merge_adjacent(pr, split_words)
        vocab.add(pr[0] + pr[1])
    return vocab


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def train_tokenizer_on_dataset(cfg: DictConfig):
    df = dataset.get_train_dataset(cfg.dataset.slice_size)
    prompts: list[str]= df["question"] + df["answer"] 
    print(prompts)
    ws: set[str] = set()
    for promp in prompts:
        ws.update(words(normalize(promp)))
    
    tokens =  bpe_train(list(ws), cfg.tokenizer.vocab_size)
    with open(cfg.tokenizer.dict, "w") as f:
        json.dump(list(tokens), f)


class Tokenizer:
    def __init__(self, tokens: list[str]):
        self.trie = PrefixSet(tokens)

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
                while is_UNK:
                    n = it.peek()
                    is_UNK = n not in self.trie
                    if is_UNK:
                        next(it)
                total.append(UNKNOWN_TOKEN)
                curr_tok = ""
                continue

            nxt = it.peek()
            if nxt is None or (curr_tok + nxt) not in self.trie:
                total.append(curr_tok)
                curr_tok = ""

    def tokenize(self, input: str) -> list[str]:
        normed = normalize(input)
        wds = words(normed)
        return [[tok for tok in self.get_tokens_for_wd(wd)] for wd in wds]

if __name__ == "__main__":
    train_tokenizer_on_dataset()