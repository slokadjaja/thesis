from utils import Params, get_model_and_hyperparams, plot_ts_with_encoding, get_dataset, get_vae_encoding, get_sax_encoding
from dataset import TSDataset
from model import VAE
import itertools
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from collections import defaultdict
from gsppy.gsp import GSP
from prefixspan import PrefixSpan
import seaborn as sns
import numpy as np
import pandas as pd
import umap
import xgboost as xgb
import shap


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


def plot_decoded_symbols(model: VAE, params: Params, plots_dir: Path):
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


def plot_tsne_umap(X_encoded, y, dataset_name: str, plots_dir: Path):
    """
    Compute and plot t-SNE and UMAP embeddings for the given dataset.

    TSNE can fail with Wine dataset, because the time series are very similar, and the representations could be all the same.
    They have then zero pairwise distance and no variance, which causes numerical error in TSNE computation (source: chatgpt)
    """
    # Compute and plot t-SNE
    X_enc_tsne = TSNE().fit_transform(X_encoded)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_enc_tsne[:, 0], X_enc_tsne[:, 1], c=y, cmap='viridis', s=16)
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.title(f"t-SNE Embeddings on {dataset_name}")
    plt.savefig(plots_dir / f"TSNE_on_{dataset_name}.png", dpi=300)
    plt.close()

    # Compute and plot UMAP
    X_enc_umap = umap.UMAP().fit_transform(X_encoded)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_enc_umap[:, 0], X_enc_umap[:, 1], c=y, cmap='viridis', s=16)
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.title(f"UMAP Embeddings on {dataset_name}")
    plt.savefig(plots_dir / f"UMAP_on_{dataset_name}.png", dpi=300)
    plt.close()


def plot_patch_groups(model_type, model_spec: str | dict, dataset: str, plots_dir: Path, plot_individual: bool = False):
    """Plot patches together that have the same encodings to see if certain patterns correspond to certain symbols."""
    
    if model_type == "vae":
        model, params = get_model_and_hyperparams(model_spec)
        train = TSDataset(dataset, "train", patch_len=params.patch_len, normalize=params.normalize,
                        norm_method=params.norm_method)
        patches = train.x
        enc = model.encode(patches).squeeze().cpu().detach().numpy()
        patches = patches.squeeze().cpu().detach().numpy()
    elif model_type == "sax":
        X_train, _, _, _ = get_dataset(dataset)
        X_encoded = get_sax_encoding(X_train, model_spec)
        enc = X_encoded.flatten()
        split_segments = [np.array_split(row, model_spec["n_segments"]) for row in X_train]
        patches = list(itertools.chain.from_iterable(split_segments))

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
            num_patches = len(patches_group)
            if model_type == "vae":
                patches_group = np.stack(patches_group, axis=0)
                mean_patch = np.mean(patches_group, axis=0)
                std_patch = np.std(patches_group, axis=0)
            elif model_type == "sax":   
                # patches could have different lengths
                max_len = max(len(patch) for patch in patches_group)
                padded_segments = np.array([np.pad(patch, (0, max_len - len(patch)), constant_values=np.nan) for patch in patches_group])
                mean_patch = np.nanmean(padded_segments, axis=0)
                std_patch = np.nanstd(padded_segments, axis=0)

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
            plt.text(0.05, 0.05, f"#patches: {num_patches}", transform=plt.gca().transAxes)

        plt.title(f"Encoding = {encoding}")
        plt.savefig(plots_dir / f"patch_group_{encoding}.png", dpi=300)
        plt.close()


def plot_global_shap_values(X_encoded, y, plots_dir: Path):
    """Plot a specific time series with its encoding."""
    le = LabelEncoder()
    y = le.fit_transform(y)
    classifier = xgb.XGBClassifier(n_estimators=100, max_depth=2).fit(X_encoded, y)

    feature_names = [str(i) for i in range(X_encoded.shape[1])]
    explainer = shap.Explainer(classifier, X_encoded, feature_names=feature_names)
    shap_values = explainer(X_encoded)

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


def plot_symbol_distributions(X_encoded, y, alphabet_size: int, plots_dir: Path, vis_method: str = "bar", 
                              group_by_class: bool = False):
    """Plot distribution of symbols at each position"""

    def _plot_distribution(encodings, filename):
        """Helper function to plot the symbol distribution based on the specified method."""
        fig, ax = plt.subplots(figsize=(10, 6))

        frequencies = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=alphabet_size), axis=0, arr=encodings
        )

        if vis_method == "heatmap":
            sns.heatmap(frequencies, ax=ax, cmap="viridis", cbar=True)
            ax.set_title("Symbol Distribution Heatmap")
            ax.set_xlabel("Position in Patch")
            ax.set_ylabel("Symbol")
            fig.tight_layout()

        elif vis_method == "bar":
            cmap = plt.get_cmap("viridis", alphabet_size)
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

    if group_by_class:
        unique_classes = np.unique(y)
        for cls in unique_classes:
            class_indices = np.where(y == cls)[0]
            class_encodings = X_encoded[class_indices]
            _plot_distribution(class_encodings, f"symbol_dist_cls{cls}_{vis_method}.png")
    else:
        _plot_distribution(X_encoded, f"symbol_dist_{vis_method}.png")


def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0


def compare_all_seq(x, y):
    """Group x by class labels in y and compute Jaccard similarity between all class pairs."""
    unique_labels = np.unique(y)  # Get unique class labels
    class_sequences = {label: x[y == label] for label in unique_labels}  # Group by class

    jaccard_scores = {}
    for label1, label2 in itertools.product(unique_labels, repeat=2):
        set1, set2 = set(class_sequences[label1]), set(class_sequences[label2])
        jaccard_scores[(label1, label2)] = jaccard_similarity(set1, set2)

    return jaccard_scores


def compare_ngram_sets(x, y, n=3, min_support=0.8):
    """
    Group x by class labels in y, compute frequent n-grams for each class, 
    and compute Jaccard similarity between all class pairs.
    
    Args:
        x (np.ndarray): Symbolic sequences (NumPy array of shape [num_samples, sequence_length]).
        y (np.ndarray): Class labels (NumPy array of shape [num_samples]).
        n (int): Minimum length of n-grams.
        min_support (float): Threshold for frequent n-grams.
        
    Returns:
        dict: Jaccard similarity scores for class pairs.
        dict: Unique frequent n-grams per class.
    """
    unique_labels = np.unique(y)  # Get unique class labels
    class_sequences = {label: x[y == label] for label in unique_labels}  # Group by class
    
    for label in unique_labels:
        print(f"label: {label}")
        print(x[y == label])

    # Compute frequent n-grams for each class
    class_ngrams = {}
    for label, sequences in class_sequences.items():
        if isinstance(sequences, np.ndarray):
            sequences = sequences.tolist()

        # frequent_sub = PrefixSpan(sequences).frequent(math.floor(min_support * len(sequences)), closed=True)
        # frequent_sub = {tuple(seq) for count, seq in frequent_sub if len(seq) >= n}   # pick sequences with n elements or longer

        # Maybe it is better if the subsequences are contiguous -> use GSP instead?
        result = GSP(sequences).search(min_support)
        frequent_sub = {seq for r in result for seq in r if len(seq) >= n}

        class_ngrams[label] = frequent_sub

    # Compute Jaccard similarities & unique frequent sequences
    jaccard_scores = {}
    unique_ngrams = {}

    for label1, label2 in itertools.product(class_ngrams.keys(), repeat=2):
        set1, set2 = class_ngrams[label1], class_ngrams[label2]
        jaccard_scores[(label1, label2)] = jaccard_similarity(set1, set2)

    for label, ngram_set in class_ngrams.items():
        unique_ngrams[label] = ngram_set - set.union(*(class_ngrams[l] for l in class_ngrams if l != label))

    return jaccard_scores, unique_ngrams


def plot_jaccard_heatmap(jaccard_scores: dict[tuple, float], labels: list, plots_dir: Path):
    """
    Plot heatmap of Jaccard similarities.
    
    Args:
        jaccard_scores: Jaccard similarity values between class pairs.
        labels : Unique class labels.
        plots_dir : File path to save the heatmap.
    """
    # Create a square matrix for heatmap
    label_to_index = {label: i for i, label in enumerate(labels)}
    jaccard_matrix = np.zeros((len(labels), len(labels)))

    # Fill the matrix with Jaccard values
    for (label1, label2), score in jaccard_scores.items():
        i, j = label_to_index[label1], label_to_index[label2]
        jaccard_matrix[i, j] = score
        jaccard_matrix[j, i] = score  # Symmetric matrix

    # Plot heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(jaccard_matrix, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Jaccard Similarity Heatmap")
    plt.savefig(plots_dir / "jaccard_heatmap.png", dpi=300, bbox_inches="tight")


def get_common_subsequences(X_encoded, y, plots_dir):
    """Use the GSP algorithm for sequential pattern mining to search for common subsequences per class"""
    
    classes = np.unique(y)
    min_support = 0.5   # fraction of ts containing the sequence

    for c in classes:
        filtered_time_series = X_encoded[y == c].tolist()
        print(filtered_time_series)
        result = PrefixSpan(filtered_time_series).frequent(math.floor(min_support * len(filtered_time_series)), closed=True)
        result = [r for r in result if len(r[1]) >= 3]   # pick sequences with 3 elements or longer
        print(f"class: {c}")
        print(result)


def main():
    model_name = "Wine_p16_a32"
    dataset = "Plane"

    plots_dir = Path(f"qualitative_plots/{model_name}_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, _, _ = get_dataset(dataset)
    X_vae = get_vae_encoding(X_train, model_name=model_name)

    get_common_subsequences(X_vae, y_train)


if __name__ == "__main__":
    main()
