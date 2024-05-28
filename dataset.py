import torch
from torch.utils.data import Dataset
import numpy as np
from utils import get_dataset_path

# for information about the datasets:  https://www.cs.ucr.edu/~eamonn/time_series_data/


class UCRDataset(Dataset):
    def __init__(self, name: str, split: str):
        """
        :param name: dataset name from ucr collection
        :param split: either "test" or "train"
        """
        arr = np.loadtxt(get_dataset_path(name, split), delimiter='\t')
        self.x = torch.from_numpy(arr[:, 1:]).unsqueeze(1).float()
        self.y = torch.from_numpy(arr[:, 0])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]
