from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from aeon.datasets import load_from_tsv_file
from aeon.transformations.collection.dictionary_based import SAX
from benchmarks.VQShape.vqshape.pretrain import LitVQShape
import json
import mlflow
from model import VAE
from typing import Any, Optional, Tuple

checkpoint_path = Path(__file__).resolve().parent / "benchmarks/VQShape/checkpoints/uea_dim256_codebook512/VQShape.ckpt"
vqshape_model = LitVQShape.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location='cpu').model


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
    data_dir = current_dir / "datasets/ucr"

    if split == "train":
        path = data_dir / f"{name}/{name}_TRAIN.tsv"
    elif split == "test":
        path = data_dir / f"{name}/{name}_TEST.tsv"
    else:
        raise Exception('Split should be either "train" or "test"')

    return path


def get_ts_length(name):
    """Get length of time series in dataset. name is either 'p2s', 'stocks' or one of UCR datasets."""
    if name == "p2s":
        ts_length = 4096
    elif name == "stocks":
        current_dir = Path(__file__).resolve().parent
        data_dir = current_dir / "datasets/stocks/nasdaq_prices.csv"
        prices_df = pd.read_csv(data_dir, index_col=0)
        ts_length = len(prices_df)
    else:
        arr = np.loadtxt(get_dataset_path(name, "train"), delimiter='\t')
        ts_length = len(arr[0]) - 1
    return ts_length


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operations have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def plot_ts_with_encoding(ts, enc, seg_len, enc_len, plot_size=(8, 4)):
    """
    :param ts: array containing time series
    :param enc: array containing time series encoded as string
    :param seg_len: length of each patch
    :param enc_len: length of encoding for each patch
    :param plot_size: size of plot
    :return: fig, ax
    """
    # todo: still need to tune some things manually
    #  (starting pos of text, text may not fit between vlines)

    fig, ax = plt.subplots()
    # fig.set_dpi(300)
    fig.set_size_inches(*plot_size)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    ax.plot(ts)

    for i in range((len(ts) // seg_len) + 1):  # loop from 0 to number of segments
        seg_x = i * seg_len
        seg_enc = enc[i * enc_len:(1 + i) * enc_len]
        seg_enc = [str(x) for x in seg_enc]
        ax.axvline(seg_x, color="k", linestyle="dashed", alpha=0.5)
        # 2, 0.05, fontsize is arbitrary, need to adjust depending on ts
        ax.text(seg_x + 2, 0.05, "".join(seg_enc), fontsize=8, transform=trans)

    plt.tight_layout()
    return fig, ax


def vae_encoding(model: VAE, data: np.ndarray):
    """Encodes entire time series datasets."""
    patch_length = model.input_dim

    # input data shape: (batch, 1, len) or (batch, len)
    if data.shape[1] == 1:
        data = data.squeeze(axis=1)

    # Pad data according to patch_length
    ts_len = data.shape[-1]
    mod = ts_len % patch_length
    if mod != 0:
        data = np.pad(data, ((0, 0), (0, patch_length - mod)), 'constant')

    data_tensor = torch.Tensor(data)
    encoded_patches = []  # array to store encodings

    for i in range(0, data_tensor.shape[-1] - patch_length + 1, patch_length):
        window = data_tensor[:, i:i + patch_length]
        window = window.unsqueeze(1)  # Shape after: (batch, 1, patch_length)
        encoded_output = model.encode(window)

        # Remove the batch dimension if needed
        # encoded_output = encoded_output.squeeze(1)

        # Store the encoded output
        encoded_patches.append(encoded_output)

    # Stack all encoded patches into a tensor
    encoded_patches = torch.cat(encoded_patches, dim=1)
    encoded_patches_np = encoded_patches.numpy()

    assert len(encoded_patches_np.shape) == 2
    assert encoded_patches_np.shape[0] == data.shape[0]

    return encoded_patches_np


def load_p2s_dataset(split: str):
    splits = {'train': 'Normal/train-00000-of-00001.parquet', 'test': 'Normal/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/AIML-TUDA/P2S/" + splits[split],
                         columns=['dowel_deep_drawing_ow', 'label'])
    X = np.array([list(row[0]) for row in df.values])
    y = pd.to_numeric(df.values[:, 1])

    return X, y


def plot_reconstructions(model, batch, n_data):
    """Plot reconstructions of given batch of patches"""
    figs = []

    for i in range(n_data):
        sample = torch.unsqueeze(batch[i], 0)

        fig, ax = plt.subplots()
        ax.plot(sample.detach().cpu().numpy().squeeze(), label="ground truth")
        ax.plot(model(sample)[1].detach().cpu().numpy().squeeze(), label="reconstruction")
        ax.legend()
        plt.title(f"Reconstruction Example {i}")

        figs.append(fig)

    return figs


def plot_loss(loss_dict: dict[str, list], dataset: str):
    """Plot loss functions given a dictionary with """
    fig, ax = plt.subplots()

    for k, v in loss_dict.items():
        ax.plot(v, alpha=0.6, label=k)

    ax.legend()
    ax.set(xlabel='Epoch', ylabel='Loss')
    plt.title(f"Loss Curve ({dataset})")

    return fig


def get_model_and_hyperparams(model_name: str, component=None) -> tuple[VAE, Params]:
    """Get a model in eval mode and hyperparameters used to train the model."""

    current_dir = Path(__file__).resolve().parent
    if component is not None:
        model_path = current_dir / "models" / model_name / component / "model.pt"
        params_path = current_dir / "models" / model_name / component / "params.json"
    else:
        model_path = current_dir / "models" / model_name / "model.pt"
        params_path = current_dir / "models" / model_name / "params.json"

    params = Params(params_path)
    input_dim = get_ts_length(params.dataset) if params.patch_len is None else params.patch_len

    vae = VAE(input_dim=input_dim, alphabet_size=params.alphabet_size, n_latent=params.n_latent,
              temperature=params.temperature, model=params.arch)
    vae.load_state_dict(
        torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    )
    vae.eval()

    return vae, params


def get_or_create_experiment(experiment_name):
    """Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist."""
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


def get_dataset(dataset: str):
    """Load datasets for downstream tasks."""
    if dataset == "p2s":
        X_train, y_train = load_p2s_dataset("train")
        X_test, y_test = load_p2s_dataset("test")
    else:
        X_train, y_train = load_from_tsv_file(get_dataset_path(dataset, "train"))
        X_test, y_test = load_from_tsv_file(get_dataset_path(dataset, "test"))

    X_train = X_train.squeeze()
    X_test = X_test.squeeze()

    return X_train, y_train, X_test, y_test


def get_sax_encoding(X, sax_params):
    sax = SAX(**sax_params)
    X_sax = sax.fit_transform(X).squeeze()
    return X_sax


def get_vae_encoding(X, model_name: str):
    """Retrieves a VAE model by name and encodes the input time series data."""
    vae, params = get_model_and_hyperparams(model_name)
    X_vae = vae_encoding(vae, X)
    return X_vae


def get_vqshape_encoding(X, params):
    X = torch.from_numpy(X).to(torch.float32)
    X = F.interpolate(X, 512, mode='linear')  # first interpolate to 512 timesteps
    X = X.squeeze()

    representations, _ = vqshape_model(X, mode='tokenize')
    tokens = representations['token']
    tokens = tokens.view(tokens.size(0), -1).detach().cpu().numpy()

    return tokens
