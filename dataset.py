import torch
from torch.utils.data import Dataset
import numpy as np
from utils import get_dataset_path

# for information about the datasets:  https://www.cs.ucr.edu/~eamonn/time_series_data/


class UCRDataset(Dataset):
    def __init__(self, name: str, split: str, patch_len=None):
        """
        :param name: dataset name from ucr collection
        :param split: either "test" or "train"
        :param patch_len: None or integer that specifies that
                            the time series should be split into chunks of length patch_len
        """
        self.patch_len = patch_len

        arr = np.loadtxt(get_dataset_path(name, split), delimiter='\t')
        x_np = arr[:, 1:]

        if self.patch_len is not None:
            # if time series length is not divisible by patch_len, remove excess values
            # todo is there a better solution?
            mod = x_np.shape[1] % self.patch_len
            if mod != 0:
                x_np = x_np[:, :-mod]

            # only self.x is split into patches
            x_np = x_np.reshape((-1, self.patch_len))

        self.x = torch.from_numpy(x_np).unsqueeze(1).float()
        self.y = torch.from_numpy(arr[:, 0])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        if self.patch_len is not None:
            return self.x[item], None
        else:
            return self.x[item], self.y[item]
