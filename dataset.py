import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from utils import get_dataset_path, load_p2s_dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from pathlib import Path
from pyts.decomposition import SingularSpectrumAnalysis

# for information about the datasets:  https://www.cs.ucr.edu/~eamonn/time_series_data/


class TSDataset(Dataset):
    def __init__(
        self,
        names: list[str] | str,
        split: str,
        patch_len=None,
        normalize=False,
        norm_method="standard",
        pad=True,
        overlap=False,
        stride=1,
        component=None,
    ):
        """
        :param names: list or string of dataset names from the UCR collection or other supported datasets
        :param split: either "test" or "train"
        :param patch_len: None or integer that specifies that
                            the time series should be split into chunks of length patch_len
        :param normalize: boolean to indicate whether data should be normalized
        :param norm_method: normalization method that will be used if normalize==True,
                            pick between "standard", "minmax", "robust"
        """
        self.patch_len = patch_len

        # Ensure names is a list
        if isinstance(names, str):
            names = [names]

        all_x = []
        all_y = []

        self.dataset_x_cumlen = []
        self.dataset_y_cumlen = []
        cumulative_length_x = 0
        cumulative_length_y = 0

        for name in names:
            if name == "p2s":
                x_np, y = load_p2s_dataset(split)
                y_tensor = torch.from_numpy(y).to(torch.int32)
            elif name == "stocks":
                current_dir = Path(__file__).resolve().parent
                data_dir = current_dir / "datasets/stocks/nasdaq_prices.csv"
                prices_df = pd.read_csv(data_dir, index_col=0)
                prices_df.index = pd.to_datetime(prices_df.index)

                # monthly returns
                returns_df = (
                    prices_df.pct_change()
                    .dropna()
                    .resample("MS")
                    .agg(lambda x: (x + 1).prod() - 1)
                )

                x_np = returns_df.T.values
                y_tensor = torch.zeros(x_np.shape[0], dtype=torch.int32)
            else:
                arr = np.loadtxt(get_dataset_path(name, split), delimiter="\t")
                y_tensor = torch.from_numpy(arr[:, 0]).to(torch.int32)
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

            if component is not None:
                ssa = SingularSpectrumAnalysis(window_size=100, groups="auto")
                X_ssa = ssa.fit_transform(x_np)

                if component == "trend":
                    x_np = X_ssa[:, 0, :]
                elif component == "seasonality":
                    x_np = X_ssa[:, 1, :]
                elif component == "residual":
                    x_np = X_ssa[:, 2, :]

            # split ts into patches
            if self.patch_len is not None:
                if not overlap:
                    mod = x_np.shape[1] % self.patch_len
                    if mod != 0:  # if time series length is not divisible by patch_len,
                        if pad:
                            # pad with zeros
                            x_np = np.pad(
                                x_np,
                                ((0, 0), (0, self.patch_len - mod)),
                                mode="constant",
                                constant_values=0,
                            )
                        else:
                            # remove excess values
                            x_np = x_np[:, :-mod]
                    # only x_np is split into patches
                    x_np = x_np.reshape((-1, self.patch_len))
                else:  # Patches should overlap
                    # Padding
                    remainder = (x_np.shape[1] - self.patch_len) % stride
                    if remainder != 0:
                        pad_len = stride - remainder
                        x_np = np.pad(
                            x_np,
                            ((0, 0), (0, pad_len)),
                            mode="constant",
                            constant_values=0,
                        )

                    patches = []
                    for ts in x_np:  # Iterate through each time series
                        # Extract patches using sliding window
                        ts_patches = [
                            list(ts[i : i + self.patch_len])
                            for i in range(0, len(ts) - self.patch_len + 1, stride)
                        ]
                        patches = patches + ts_patches
                    x_np = np.array(patches)

            all_x.append(torch.from_numpy(x_np).unsqueeze(1).float())
            all_y.append(y_tensor)

            cumulative_length_x += len(x_np)
            cumulative_length_y += len(y_tensor)
            self.dataset_x_cumlen.append(cumulative_length_x)
            self.dataset_y_cumlen.append(cumulative_length_y)

        # Combine all datasets
        self.x = torch.cat(all_x, dim=0)
        self.y = torch.cat(all_y, dim=0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        if self.patch_len is not None:  # Need to find y index for 'item'
            # Determine which dataset 'item' belongs to
            dataset_idx = next(
                idx for idx, length in enumerate(self.dataset_x_cumlen) if item < length
            )

            # Calculates the relative index of 'item' within the specific dataset
            if dataset_idx == 0:
                dataset_x_length = self.dataset_x_cumlen[dataset_idx]
                dataset_y_length = self.dataset_y_cumlen[dataset_idx]
                relative_x = item
            else:
                dataset_x_length = (
                    self.dataset_x_cumlen[dataset_idx]
                    - self.dataset_x_cumlen[dataset_idx - 1]
                )
                dataset_y_length = (
                    self.dataset_y_cumlen[dataset_idx]
                    - self.dataset_y_cumlen[dataset_idx - 1]
                )
                relative_x = item - self.dataset_x_cumlen[dataset_idx - 1]

            relative_y = relative_x // (dataset_x_length // dataset_y_length)
            y_idx = (
                relative_y
                if dataset_idx == 0
                else relative_y + self.dataset_y_cumlen[dataset_idx - 1]
            )

            return self.x[item], self.y[y_idx]

            # series_idx = item // (self.x.shape[0] // len(self.y))
            # return self.x[item], self.y[series_idx]
        else:
            return self.x[item], self.y[item]
