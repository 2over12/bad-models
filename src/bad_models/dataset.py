from datasets import load_dataset, Dataset

DATASET_NAME = "microsoft/orca-math-word-problems-200k"
def get_train_dataset(slc:int) -> Dataset:
    ds = load_dataset(DATASET_NAME, split="train")
    
    return ds[0: slc]