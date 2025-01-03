"""
Score when dataset is saved
"""

from datasets import load_dataset
from prm import config
from src.prm.utils.score import score

dataset = load_dataset("json", data_files="./output/completions.json", split="train")
dataset = score(dataset, config)
