"""Compare performance of TS classification using SAX and VAE encodings"""

from aeon.transformations.collection.dictionary_based import SAX
from aeon.datasets import load_from_tsv_file
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from utils import *
from model import VAE
import torch

model_path = "../baseline_models/fc/model.pt"
params_path = "../baseline_models/fc/params.json"


def vae_encoding(model, data, patch_length):
    # assume data shape: (batch, 1, len)
    # todo: put this method in model class or utils?

    data = data.squeeze()

    # pad data according to patch_length
    ts_len = data.shape[-1]
    mod = ts_len % patch_length
    data = np.pad(data, ((0, 0), (0, patch_length - mod)), mode='constant', constant_values=0)

    data_tensor = torch.Tensor(data)
    encoded_patches = []    # array to store encodings

    for i in range(0, data_tensor.shape[-1] - patch_length + 1, patch_length):
        window = data_tensor[:, i:i + patch_length]
        window = window.unsqueeze(1)  # Shape: (batch, 1, patch_length)
        encoded_output = model.encode(window)

        # Remove the batch dimension if needed
        encoded_output = encoded_output.squeeze(1)  # Shape: (encoded_dim)

        # Store the encoded output
        encoded_patches.append(encoded_output)

    # Stack all encoded patches into a tensor
    encoded_patches = torch.cat(encoded_patches, dim=1)
    encoded_patches_np = encoded_patches.numpy()

    return encoded_patches_np


def decision_tree_classifier(X_train, y_train, X_test, y_test):
    # Fit decision tree classifier
    clf = DecisionTreeClassifier().fit(X_train, y_train)

    # Calculate F1 and accuracy score
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Confusion matrix:\n{cm}")
    print(f"F1 score: {f1:.2f}")
    print(f"Accuracy: {acc:.2f}")

    return {"f1": f1, "accuracy": acc}


def main():
    params = Params(params_path)

    # todo do we need normalize, norm method (??)
    dataset, patch_len, alphabet_size, n_latent, arch = \
        params.dataset, params.patch_len, params.alphabet_size, params.n_latent, params.arch

    # Define parameters and paths
    # todo make sure vae and sax use same n_segments, alphabet_size
    n_segments_sax = 128
    alphabet_size_sax = 8

    # Load datasets
    X_train, y_train = load_from_tsv_file(get_dataset_path(dataset, "train"))
    X_test, y_test = load_from_tsv_file(get_dataset_path(dataset, "test"))

    # Get SAX symbols
    sax = SAX(n_segments=n_segments_sax, alphabet_size=alphabet_size_sax)
    X_train_sax = sax.fit_transform(X_train).squeeze()
    X_test_sax = sax.fit_transform(X_test).squeeze()

    # Get vae encodings
    vae = VAE(patch_len, alphabet_size, n_latent, arch)
    vae.load_state_dict(torch.load(model_path))
    vae.eval()

    X_train_vae = vae_encoding(vae, X_train, patch_len)
    X_test_vae = vae_encoding(vae, X_test, patch_len)

    print(f"Dataset: {dataset}")
    print("SAX results:")
    decision_tree_classifier(X_train_sax, y_train, X_test_sax, y_test)

    print("VAE results:")
    decision_tree_classifier(X_train_vae, y_train, X_test_vae, y_test)


if __name__ == "__main__":
    main()
