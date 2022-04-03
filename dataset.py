from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import os


class Quark(Dataset):
    def __init__(self, path, name1):
        super(Quark, self).__init__()
        name1 = os.path.join(path, name1)
        df1 = pd.read_parquet(name1, engine='fastparquet')
        df1.drop(columns='X_jets', inplace=True)
        self.df = np.array(df1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        data = self.df[item]
        return torch.from_numpy(np.array(data[:-1])), torch.Tensor(np.array(data[-1]))
