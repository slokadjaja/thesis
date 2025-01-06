from utils import get_model_and_hyperparams, get_dataset_path, load_p2s_dataset, vae_encoding, plot_ts_with_encoding
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from aeon.datasets import load_from_tsv_file
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import pandas as pd

# code from decoder_test.ipynb, plot_test.ipynb, ppt.ipynb


def get_dataset(dataset: str):
    # Load datasets
    if dataset == "p2s":
        X_train, y_train = load_p2s_dataset("train")
        X_test, y_test = load_p2s_dataset("test")
    else:
        X_train, y_train = load_from_tsv_file(get_dataset_path(dataset, "train"))
        X_test, y_test = load_from_tsv_file(get_dataset_path(dataset, "test"))

    X_train = X_train.squeeze()
    X_test = X_test.squeeze()

    return X_train, y_train, X_test, y_test


def plot_dataset_classes(dataset: str):
    """Plot time series from a specified dataset, where time series are grouped by class."""
    X_train, y_train, _, _ = get_dataset(dataset)

    df = pd.DataFrame(X_train).T
    df.columns = y_train  # Assign column names as labels
    df = (df
          .reset_index()
          .melt(id_vars='index', var_name='Label', value_name='Value')
          .rename(columns={'index': 'Time'})
          )

    fig = sns.lineplot(data=df, x='Time', y='Value', hue='Label').set_title(dataset).get_figure()
    fig.savefig(f"{dataset}.png", dpi=300)


def plot_decoded_symbols(model, params, plots_dir, plot_random=False):
    for i in range(params.n_latent):  # iterate over positions in encoding
        plt.figure()
        for j in range(params.alphabet_size):  # iterate over possible values for position
            enc = torch.zeros(params.n_latent, dtype=torch.int64)
            enc[i] = j
            z = torch.unsqueeze(F.one_hot(enc, num_classes=params.alphabet_size),
                                0)  # z.shape: (1, n_latent, alphabet_size)
            output = model.decoder(z.float())
            plt.plot(output.detach().numpy().squeeze(), label=f'val={j}')

        plt.title(f'pos: {i}')
        plt.savefig(plots_dir / 'decoded_symbols.png', dpi=300)

    if plot_random:
        # generate random
        for i in range(20):
            enc = torch.randint(0, params.alphabet_size - 1, (params.n_latent,))
            print(enc)  # todo use label inside plot
            z = torch.unsqueeze(F.one_hot(enc, num_classes=params.alphabet_size),
                                0)  # z.shape: (1, n_latent, alphabet_size)
            output = model.decoder(z.float())
            plt.plot(output.detach().numpy().squeeze(), label=i)
            plt.savefig(plots_dir / 'random_decoded_symbols.png', dpi=300)


def plot_tsne(model, params, dataset, plots_dir):
    # get data
    X_train, y_train, X_test, y_test = get_dataset(dataset)
    X_train_vae = vae_encoding(model, X_train, params.patch_len)

    X_train_vae_embedded = TSNE().fit_transform(X_train_vae)
    plt.scatter(X_train_vae_embedded[:, 0], X_train_vae_embedded[:, 1], c=y_train)

    # todo umap
    # todo save fig

def plot_ts_with_encodings(model, params, dataset, plots_dir):
    # get data
    X_train, y_train, X_test, y_test = get_dataset(dataset)
    X_train_vae = vae_encoding(model, X_train, params.patch_len)

    idx = 0
    plot_ts_with_encoding(X_train[idx], X_train_vae[idx], params.patch_len, params.n_latent)
    # todo save fig
    # todo plot class label + prediction, use shap to plot feature importance for prediction

def plot_cls_feature_importance(model, params, dataset, plots_dir):
    # get data
    X_train, y_train, X_test, y_test = get_dataset(dataset)
    X_train_vae = vae_encoding(model, X_train, params.patch_len)
    # X_test_vae = vae_encoding(model, X_test, params.patch_len)

    clf = DecisionTreeClassifier().fit(X_train_vae, y_train)
    feature_importance = clf.feature_importances_

    # todo overlay feature importance on ts plot + encodings
    #   can maybe compare with sax?


# todo use sequential pattern mining to search for common subsequences per class?
# todo error analysis:
#   Review cases where the classifier made incorrect predictions, focusing on the symbolic encoding and comparing it to
#   the original time series.
#   Use SHAP or LIME to identify which symbolic features were most influential in the incorrect predictions.
#   This can help determine if specific symbols or subsequences lead to errors.
# todo verify if certain symbols correspond to certain patterns (both ways)
# todo test models with its training dataset and unseen dataset
# todo Select pairs of time series with known relationships (e.g., similar trends, different noise levels).
#   Measure distances in the latent space and verify if they correspond to their semantic similarity.

def main():
    model_name = "Wine_p16_a32"
    dataset = "Wine"

    plots_dir = Path(f"qualitative_plots/{model_name}_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    vae, params = get_model_and_hyperparams(model_name)

    plot_decoded_symbols(vae, params, plots_dir)
    plot_tsne(vae, params, dataset, plots_dir)
    plot_ts_with_encodings(vae, params, dataset, plots_dir)
    plot_cls_feature_importance(vae, params, dataset, plots_dir)

if __name__ == "__main__":
    main()
