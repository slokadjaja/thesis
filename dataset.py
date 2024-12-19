import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from utils import get_dataset_path, load_p2s_dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# for information about the datasets:  https://www.cs.ucr.edu/~eamonn/time_series_data/

# todo change class name
class UCRDataset(Dataset):
    def __init__(self, name: str, split: str, patch_len=None, normalize=False, norm_method="standard", pad=True,
                 overlap=False, stride=1):
        """
        :param name: dataset name from ucr collection
        :param split: either "test" or "train"
        :param patch_len: None or integer that specifies that
                            the time series should be split into chunks of length patch_len
        :param normalize: boolean to indicate whether data should be normalized
        :param norm_method: normalization method that will be used if normalize==True,
                            pick between "standard", "minmax", "robust"
        """
        self.patch_len = patch_len

        if name == "p2s":
            x_np, y = load_p2s_dataset(split)
            self.y = torch.from_numpy(y).to(torch.int32)
        elif name == "stocks":
            prices_df = pd.read_csv("datasets/stocks/nasdaq_prices.csv", index_col=0)
            prices_df.index = pd.to_datetime(prices_df.index)

            # monthly returns
            returns_df = prices_df.pct_change().dropna().resample('MS').agg(lambda x: (x + 1).prod() - 1)

            x_np = returns_df.T.values
            self.y = torch.zeros(x_np.shape[0], dtype=torch.int32)
        else:
            arr = np.loadtxt(get_dataset_path(name, split), delimiter='\t')
            self.y = torch.from_numpy(arr[:, 0]).to(torch.int32)
            x_np = arr[:, 1:]

        # normalize samples
        if normalize:
            if norm_method == "standard":
                scaler = StandardScaler()
            elif norm_method == "minmax":
                scaler = MinMaxScaler()
            elif norm_method == "robust":
                scaler = RobustScaler()
            else:
                raise Exception("choose between standard, minmax, or robust")

            x_np = scaler.fit_transform(x_np)

        # split ts into patches
        if self.patch_len is not None:
            if not overlap:
                mod = x_np.shape[1] % self.patch_len
                if mod != 0:    # if time series length is not divisible by patch_len,
                    if pad:
                        # pad with zeros
                        x_np = np.pad(x_np, ((0, 0), (0, patch_len-mod)), mode='constant', constant_values=0)
                    else:
                        # remove excess values
                        x_np = x_np[:, :-mod]
                # only self.x is split into patches
                x_np = x_np.reshape((-1, self.patch_len))
            else:   # Patches should overlap
                # Padding
                remainder = (x_np.shape[1] - self.patch_len) % stride
                if remainder != 0:
                    pad_len = stride - remainder
                    x_np = np.pad(x_np, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)

                patches = []
                for ts in x_np:  # Iterate through each time series
                    # Extract patches using sliding window
                    ts_patches = [list(ts[i:i + patch_len]) for i in range(0, len(ts) - patch_len + 1, stride)]
                    patches = patches + ts_patches
                x_np = np.array(patches)

        self.x = torch.from_numpy(x_np).unsqueeze(1).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        if self.patch_len is not None:
            series_idx = item // (self.x.shape[0] // len(self.y))
            return self.x[item], self.y[series_idx]
        else:
            return self.x[item], self.y[item]
