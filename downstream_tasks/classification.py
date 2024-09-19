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

# params needed:
# dataset
# arch
# n_segments, alphabet size -> for sax
# patch len -> padding/deleting?
# normalize, norm method (??)


def vae_encoding(model, data, patch_length):
    # todo: put this method in model class?

    data_tensor = torch.from_numpy(data)

    # pad data according to patch_length
    ts_len = data.shape[-1]

    # array/tensor to store encodings
    
    # apply vae encode method

    return torch.tensor(0)


def decision_tree_classifier(X_train, y_train, X_test, y_test):
    # Fit decision tree classifier
    clf = DecisionTreeClassifier().fit(X_train, y_train)

    # Calculate F1 and accuracy score
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Confusion matrix: {cm}")
    print(f"F1 score: {f1:.2f}")
    print(f"Accuracy: {acc:.2f}")

    return {"f1": f1, "accuracy": acc}


def main():
    params = Params(params_path)

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

    # todo test vae_encoding method
    X_train_vae = vae_encoding(vae, X_train, patch_len)

    # X_test_vae = vae_encoding(vae, X_test, patch_len)
    #
    # print("SAX results:")
    # decision_tree_classifier(X_train_sax, y_train, X_test_sax, y_test)
    #
    # print("VAE results:")
    # decision_tree_classifier(X_train_vae, y_train, X_test_vae, y_test)


if __name__ == "__main__":
    main()
