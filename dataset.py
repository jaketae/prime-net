import json
import os

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
            return self.primes[idx], 1
        return self.composites[idx - len_primes], 0

    def __len__(self):
        return len(self.primes) + len(self.composites)
