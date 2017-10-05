#!/usr/bin/env python3.5

import os
import numpy as np
from datetime import datetime


def getCurrentTime(time=True, date=False):
    now = datetime.now()
    s = ""
    if date:
        s += "{} ".format(now.date())
    if time:
        s += "{:02d}:{:02d}:{:02d}".format(now.hour, now.minute, now.second)
    return s.strip()

def load_data(dataset, filename_out="lumps"):
    if dataset == "lumps":
        path = "./dataset/"
    else:
        raise KeyError("Unknown dataset: {}".format(dataset))
    files = sorted(os.listdir(path))
    num_files = len(files)
    percent = 2
    split_distance = num_files * percent // 100
    print("Loading {} samples from dataset {}".format(num_files, dataset))
    for i, filename in enumerate(files):
        with open(path + filename, "r") as f:
            content = f.read()
            sample = np.array([[float(n) for n in line.split(",")] for line in content.split("\n") if len(line) > 0])
            try:
                data = np.concatenate((data, [sample]))
            except NameError:
                data = np.array([sample])
        if (i + 1) % split_distance == 0:
            print("{}. {}% loaded".format(getCurrentTime(), (i + 1) * 100 // num_files))
    print("Shape:", data.shape)
    np.save(filename_out, data)


if __name__ == "__main__":
    load_data("lumps")
