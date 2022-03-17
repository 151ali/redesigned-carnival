import numpy as np
import torch
from torch.utils.data import Dataset
import sys
import os

sys.path.extend(['../'])
from feeders import tools


class Feeder(Dataset):
    def __init__(self, root_dir,
        annotations_file = "annotations.npy",
        data_file        = "data.npy"
    ):
        self.root_dir = root_dir
        self.annotations = np.load(
            os.path.join(root_dir, annotations_file)
        )
        self.data = self.annotations = np.load(
            os.path.join(root_dir, data_file)
        )

    def __len__(self):
        return len(self.annotations)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.annotations[index]
        
        return data_numpy, label, index


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    print("hello world")
    print(sys.path[0])
    d = Feeder("/path/to/data/")
    print(len(d))