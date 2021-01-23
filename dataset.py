import json
import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset


class NumberDataset(Dataset):
    def __init__(self, mode):
        directory = os.path.join("data", mode)
        with open(os.path.join(directory, "primes.txt")) as p_file:
            self.primes = json.load(p_file)
        with open(os.path.join(directory, "composites.txt")) as c_file:
            self.composites = json.load(c_file)

    def __getitem__(self, idx):
        len_primes = len(self.primes)
        if idx < len_primes:
            label = 1.0
            data = self.primes[idx]
        else:
            label = 0.0
            data = self.composites[idx - len_primes]
        return torch.tensor([int(digit) for digit in data]), label

    def __len__(self):
        return len(self.primes) + len(self.composites)


def get_pos_weight():
    dataset = NumberDataset("train")
    num_primes = len(dataset.primes)
    num_composites = len(dataset.composites)
    return torch.tensor(num_composites / num_primes)


def make_loader(mode, batch_size=32):
    assert mode in {
        "train",
        "val",
        "test",
    }, "`mode` must be one of 'train', 'val', or 'test'"
    dataset = NumberDataset(mode)
    shuffle = mode == "train"
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=shuffle
    )
    return data_loader
