from utils import get_model_and_hyperparams, vae_encoding, plot_ts_with_encoding, get_dataset
from dataset import TSDataset
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from collections import defaultdict
import seaborn as sns
import numpy as np
import pandas as pd
import umap
import xgboost as xgb
import shap

# todo plot_tsne_umap, plot_patch_groups, plot_global_shap_values, plot_symbol_distributions -> should also work with sax


def find_sequence_in_pattern(sequence, pattern):
    """
    Locate indices in the sequence that match the pattern. Pattern could be non-contiguous

    Parameters:
        sequence (list): The main sequence to search in.
        pattern (list): The pattern to match.

    Returns:
        list: List of indices in the sequence that match the pattern.
              Returns an empty list if no match is found.
    """
    indices = []
    seq_idx = 0  # Pointer for the sequence

    # Iterate through the pattern
    for pat_item in pattern:
        # Search for the current pattern item in the sequence from the current pointer
        while seq_idx < len(sequence) and sequence[seq_idx] != pat_item:
            seq_idx += 1
        
        # If the item is found, save the index
        if seq_idx < len(sequence):
            indices.append(seq_idx)
            seq_idx += 1  # Move to the next position in the sequence
        else:
            # Pattern item not found, return an empty list
            return []

    return indices


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


def plot_decoded_symbols(model, params, plots_dir):
    """Generate and save plots of the decoded output for each possible symbol at each latent position."""
    cmap = plt.get_cmap("viridis", params.alphabet_size)

    for pos in range(params.n_latent):  # iterate over positions in encoding
        plt.figure(figsize=(8, 6))
        for sym in range(params.alphabet_size):  # iterate over possible values for position
            enc = torch.zeros(params.n_latent, dtype=torch.int64)
            enc[pos] = sym  # set symbol at position 'pos'
            z = torch.unsqueeze(F.one_hot(enc, num_classes=params.alphabet_size),
                                0)  # z.shape: (1, n_latent, alphabet_size)
            output = model.decoder(z.float()).detach().numpy().squeeze()    # decode latent vector
            plt.plot(output, label=sym, color=cmap(sym))

        plt.title(f'Decoded Outputs for Latent Position: {pos}')
        plt.legend(loc='center right', prop={'size': 8}, bbox_to_anchor=(1.08, 0.5), title="Symbol")
        plt.tight_layout()
        plt.savefig(plots_dir / f'decoded_symbols_pos_{pos}.png', dpi=300)
        plt.close()


def plot_tsne_umap(model, params, dataset, plots_dir):
    """
    Compute and plot t-SNE and UMAP embeddings for the given dataset.

    TSNE can fail with Wine dataset, because the time series are very similar, and the representations could be all the same.
    They have then zero pairwise distance and no variance, which causes numerical error in TSNE computation (chatgpt)
    """

    # Load dataset and get VAE encodings
    X_train, y_train, _, _ = get_dataset(dataset)
    X_train_vae = vae_encoding(model, X_train)

    # Compute and plot t-SNE
    X_train_vae_tsne = TSNE().fit_transform(X_train_vae)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_train_vae_tsne[:, 0], X_train_vae_tsne[:, 1], c=y_train, cmap='viridis', s=16)
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.title(f"t-SNE Embeddings on {dataset}")
    plt.savefig(plots_dir / f"TSNE_on_{dataset}.png", dpi=300)
    plt.close()

    # Compute and plot UMAP
    X_train_vae_umap = umap.UMAP().fit_transform(X_train_vae)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_train_vae_umap[:, 0], X_train_vae_umap[:, 1], c=y_train, cmap='viridis', s=16)
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.title(f"UMAP Embeddings on {dataset}")
    plt.savefig(plots_dir / f"UMAP_on_{dataset}.png", dpi=300)
    plt.close()


def plot_patch_groups(model, params, dataset, plots_dir, plot_individual=False):
    """Plot patches together that have the same encodings to see if certain patterns correspond to certain symbols."""
    train = TSDataset(dataset, "train", patch_len=params.patch_len, normalize=params.normalize,
                      norm_method=params.norm_method)

    patches = train.x
    enc = model.encode(patches).squeeze().cpu().detach().numpy()
    patches = patches.squeeze().cpu().detach().numpy()

    # Group patches by their encodings
    grouped_patches = defaultdict(list)
    for patch, encoding in zip(patches, enc):
        if isinstance(encoding, np.ndarray) or isinstance(encoding, list):
            encoding_key = tuple(encoding)
        else:
            encoding_key = encoding
        grouped_patches[encoding_key].append(patch)

    # Create plots for each group
    for idx, (encoding, patches_group) in enumerate(grouped_patches.items()):
        plt.figure(figsize=(8, 6))

        if plot_individual:
            for patch in patches_group:
                plt.plot(patch, alpha=0.5)  # Overlay patches with some transparency
        else:
            patches_group = np.stack(patches_group, axis=0)
            mean_patch = np.mean(patches_group, axis=0)
            std_patch = np.std(patches_group, axis=0)

            plt.plot(mean_patch, label="Mean", color="blue")
            plt.fill_between(
                range(len(mean_patch)),
                mean_patch - std_patch,
                mean_patch + std_patch,
                color="blue",
                alpha=0.2,
                label="Mean ± 1 std"
            )
            plt.legend()

        plt.title(f"Encoding = {encoding}")
        plt.savefig(plots_dir / f"patch_group_{encoding}.png", dpi=300)
        plt.close()


def plot_global_shap_values(model, params, dataset, plots_dir):
    """Plot a specific time series with its encoding."""

    X_train, y_train, X_test, y_test = get_dataset(dataset)
    X_train_vae = vae_encoding(model, X_train)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    classifier = xgb.XGBClassifier(n_estimators=100, max_depth=2).fit(X_train_vae, y_train)

    feature_names = [str(i) for i in range(X_train_vae.shape[1])]
    explainer = shap.Explainer(classifier, X_train_vae, feature_names=feature_names)
    shap_values = explainer(X_train_vae)

    # multiclass classification results in multiple shap values for each feature
    if len(shap_values.values.shape) == 3:
        shap_values.values = shap_values.values.mean(axis=2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot the first SHAP bar plot (mean absolute SHAP values)
    shap.plots.bar(shap_values, show=False, ax=axes[0])
    axes[0].set_title("SHAP Values (Mean)")

    # Plot the second SHAP bar plot (max absolute SHAP values)
    shap.plots.bar(shap_values.abs.max(0), show=False, ax=axes[1])
    axes[1].set_title("SHAP Values (Max)")

    plt.tight_layout()
    plt.savefig(plots_dir / "global_shap_values.png", dpi=300)


def plot_symbol_distributions(model, params, dataset, plots_dir, vis_method="bar", group_by_class=False,
                              split="train"):
    """Plot distribution of symbols at each position"""

    def _plot_distribution(encodings, filename):
        """Helper function to plot the symbol distribution based on the specified method."""
        fig, ax = plt.subplots(figsize=(10, 6))

        frequencies = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=params.alphabet_size), axis=0, arr=encodings
        )

        if vis_method == "heatmap":
            sns.heatmap(frequencies, ax=ax, cmap="viridis", cbar=True)
            ax.set_title("Symbol Distribution Heatmap")
            ax.set_xlabel("Position in Patch")
            ax.set_ylabel("Symbol")
            fig.tight_layout()

        elif vis_method == "bar":
            cmap = plt.get_cmap("viridis", params.alphabet_size)
            positions = list(range(encodings.shape[1]))
            bottom = np.zeros(encodings.shape[1])
            for symbol, freq in enumerate(frequencies):
                ax.bar(positions, freq, bottom=bottom, label=symbol, color=cmap(symbol))
                bottom += freq

            ax.legend(loc='center right', prop={'size': 8}, bbox_to_anchor=(1.1, 0.5), title="Symbol")
            ax.set_title("Symbol Distribution Bar Chart")
            ax.set_xlabel("Position in Patch")
            ax.set_ylabel("Frequency")
        else:
            raise ValueError(f"Unsupported visualization method: {vis_method}")

        fig.savefig(plots_dir / filename, dpi=300)
        plt.close(fig)

    # Load dataset and get VAE encodings
    X_train, y_train, X_test, y_test = get_dataset(dataset)
    if split == "train":
        X, y = X_train, y_train
    elif split == "test":
        X, y = X_test, y_test
    else:
        raise Exception("Invalid split")

    X_vae = vae_encoding(model, X)

    if group_by_class:
        unique_classes = np.unique(y)
        for cls in unique_classes:
            class_indices = np.where(y == cls)[0]
            class_encodings = X_vae[class_indices]
            _plot_distribution(class_encodings, f"symbol_dist_cls{cls}_{vis_method}.png")
    else:
        _plot_distribution(X_vae, f"symbol_dist_{vis_method}.png")


def plot_ts_with_encodings(model, params, dataset, plots_dir, idx):
    """Plot a specific time series with its encoding."""
    X_train, y_train, X_test, y_test = get_dataset(dataset)
    X_train_vae = vae_encoding(model, X_train, params.patch_len)

    fig, ax = plot_ts_with_encoding(X_train[idx], X_train_vae[idx], params.patch_len, params.n_latent)
    fig.savefig(plots_dir / f"ts_with_encodings_{idx}.png", dpi=300)

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
#
# todo error analysis:
#   Review cases where the classifier made incorrect predictions, focusing on the symbolic encoding and comparing it to
#   the original time series.
#   Use SHAP or LIME to identify which symbolic features were most influential in the incorrect predictions.
#   This can help determine if specific symbols or subsequences lead to errors.
#   see notion experiment page
#
# todo test models with its training dataset and unseen dataset
# todo Select pairs of time series with known relationships (e.g., similar trends, different noise levels).
#   Measure distances in the latent space and verify if they correspond to their semantic similarity.


def main():
    model_name = "Wine_p16_a32"
    dataset = "ArrowHead"

    plots_dir = Path(f"qualitative_plots/{model_name}_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    vae, params = get_model_and_hyperparams(model_name)
    plot_patch_groups(vae, params, dataset, plots_dir)

if __name__ == "__main__":
    main()
