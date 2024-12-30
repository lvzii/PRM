import datasets
from datasets import Dataset, load_dataset


def get_dataset():
    dataset = load_dataset("json", data_files="data_eda/rw-v7-test.json")
    return dataset


def save_dataset():
    pass
