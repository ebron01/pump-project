import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    # def __init__(self, dataframe, target, features, sequence_length=5):
    def __init__(self, features, target,  sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        # self.y = torch.tensor(self.target.values).float()
        # self.X = torch.tensor(self.features.values).float()
        self.X = torch.tensor(np.asarray(self.features)).float()
        self.y = torch.tensor(np.asarray(self.target)).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]
