from pathlib import Path
import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import json


class Params:
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def get_dataset_path(name, split):
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir / "datasets_ucr"

    if split == "train":
        path = data_dir / f"{name}/{name}_TRAIN.tsv"
    elif split == "test":
        path = data_dir / f"{name}/{name}_TEST.tsv"
    else:
        raise Exception('Split should be either "train" or "test"')

    return path


def get_ts_length(name):
    arr = np.loadtxt(get_dataset_path(name, "train"), delimiter='\t')
    return len(arr[0]) - 1


def sample_gumbel(shape: torch.Size, eps=1e-20) -> torch.Tensor:
    """ Samples from the Gumbel distribution given a tensor shape and value of epsilon for numerical stability """
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """ Adds Gumbel noise to `logits` and applies softmax along the last dimension """
    input_shape = logits.shape
    y = logits + sample_gumbel(logits.shape)
    return torch.nn.functional.softmax(y / temperature, dim=-1).view(input_shape)


def cat_kl_div(logits, n_latent, alphabet_size):
    q = dist.Categorical(logits=logits)
    p = dist.Categorical(probs=torch.full((n_latent, alphabet_size), 1.0/alphabet_size))
    kl = dist.kl.kl_divergence(q, p)
    return torch.mean(torch.sum(kl, dim=1))


def reconstruction_loss(x_true, x_out):
    return F.mse_loss(x_true, x_out)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():   # GPU operations have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def plot_ts_with_encoding(ts, enc, seg_len, enc_len):
    """
    :param ts: array containing time series
    :param enc: array containing time series encoded as string
    :param seg_len: length of each patch
    :param enc_len: length of encoding for each patch
    :return: fig, ax
    """
    # todo: still need to tune some things manually
    #  (starting pos of text, text may not fit between vlines)

    fig, ax = plt.subplots()
    fig.set_dpi(300)
    fig.set_size_inches(8, 4)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    ax.plot(ts)

    for i in range((len(ts) // seg_len) + 1):   # loop from 0 to number of segments
        seg_x = i * seg_len
        seg_enc = enc[i * enc_len:(1 + i) * enc_len]
        seg_enc = [str(x) for x in seg_enc]
        ax.axvline(seg_x, color="k", linestyle="dashed", alpha=0.5)
        # 2, 0.05, fontsize is arbitrary, need to adjust depending on ts
        ax.text(seg_x + 2, 0.05, "".join(seg_enc), fontsize=8, transform=trans)

    plt.tight_layout()
    return fig, ax


def conv_out_len_multiple(l_in, out_channels, arr):
    # arr: contain tuples (kernel, stride, pad)
    out_len_arr = [l_in]
    for a in arr:
        out_len_arr.append(conv_out_len(out_len_arr[-1], *a))

    return out_len_arr[-1] * out_channels


def conv_out_len(l_in, kernel, stride, pad=0):
    import math
    return math.floor(1 + (l_in + 2 * pad - kernel) / stride)
