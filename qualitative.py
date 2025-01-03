from utils import get_model_and_hyperparams, get_dataset_path, load_p2s_dataset, vae_encoding, plot_ts_with_encoding
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from aeon.datasets import load_from_tsv_file
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier

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

def plot_cls_feature_importance(model, params, dataset, plots_dir):
    # get data
    X_train, y_train, X_test, y_test = get_dataset(dataset)
    X_train_vae = vae_encoding(model, X_train, params.patch_len)
    # X_test_vae = vae_encoding(model, X_test, params.patch_len)

    clf = DecisionTreeClassifier().fit(X_train_vae, y_train)
    feature_importance = clf.feature_importances_

    # todo overlay feature importance on ts plot + encodings


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
