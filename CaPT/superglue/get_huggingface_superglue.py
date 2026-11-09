import os
import sys
from datasets import load_dataset
import pickle

dataset_names = ['boolq', 'cb', 'copa', 'multirc', 'rte', 'wic']

for dataset in dataset_names:
    os.makedirs(f"data/superglue/{dataset}", exist_ok=True)
    data = load_dataset('super_glue', dataset)
    for key in data.keys():
        path = f"data/superglue/{dataset}/{key}.pkl"
        pickle.dump(data[key], open(path, 'wb'))


