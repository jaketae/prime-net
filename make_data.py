import argparse
import os

import numpy as np
import pandas as pd
import sympy
from sklearn.model_selection import train_test_split


def main(args):
    max_num, val_size, test_size = args.max_num, args.val_size, args.test_size
    all_nums = set(range(2, max_num))
    primes = list(sympy.primerange(2, max_num))
    composites = list(all_nums - set(primes))
    composites.sort()
    for p_nums, c_nums, mode in zip(
        split(primes, val_size, test_size),
        split(composites, val_size, test_size),
        ("train", "val", "test"),
    ):
        directory = os.path.join(os.getcwd(), "data", mode)
        if not os.path.exists(directory):
            os.mkdir(directory)
        pd.DataFrame(p_nums).to_csv(os.path.join(directory, "primes.csv"))
        pd.DataFrame(c_nums).to_csv(os.path.join(directory, "composites.csv"))


def split(nums, val_size, test_size):
    train_size = 1 - val_size - test_size
    train, tv = train_test_split(nums, train_size=train_size, shuffle=False)
    proportion = val_size / (val_size + test_size)
    val, test = train_test_split(tv, train_size=proportion, shuffle=True)
    return train, val, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_num", type=int, default=100000)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
