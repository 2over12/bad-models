import pandas as pd
import more_itertools
import dataset


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


def train_tokenizer_on_dataset():
    df = dataset.get_train_dataset()
    prompts: list[str]= df["question"] + df["answer"]
    prompts = prompts 
    ws: set[str] = set()
    for promp in prompts:
        ws.update(words(normalize(promp)))
    
    return bpe_train(list(ws), 400)

if __name__ == "__main__":
    print(train_tokenizer_on_dataset())