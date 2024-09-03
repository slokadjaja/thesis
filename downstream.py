""" Downstream tasks to compare results with benchmarks """

from aeon.transformations.collection.dictionary_based import SAX
from aeon.datasets import load_from_tsv_file
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
from utils import get_dataset_path

model_path = "baseline_models/fc/model.pt"


def vae_encoding():
    # todo
    pass


# Implemented so far: TS classification using decision tree, aim to compare SAX and VAE encodings
if __name__ == "__main__":
    # Define parameters and paths
    n_segments = 128
    alphabet_size = 8

    # Load datasets
    dataset = "ArrowHead"
    X_train, y_train = load_from_tsv_file(get_dataset_path(dataset, "train"))
    X_test, y_test = load_from_tsv_file(get_dataset_path(dataset, "test"))

    # Get SAX symbols
    sax = SAX(n_segments=n_segments, alphabet_size=alphabet_size)
    X_train_sax = sax.fit_transform(X_train).squeeze()
    X_test_sax = sax.fit_transform(X_test).squeeze()

    # Get vae encodings
    # todo

    # Fit decision tree classifier
    clf = DecisionTreeClassifier().fit(X_train_sax, y_train)

    # Calculate F1 and accuracy score
    y_pred = clf.predict(X_test_sax)
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix: {cm}")
    print(f"F1 score: {f1:.2f}")
    print(f"Accuracy: {acc:.2f}")
