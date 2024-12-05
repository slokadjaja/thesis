"""Compare performance of TS classification using SAX and VAE encodings"""

from aeon.transformations.collection.dictionary_based import SAX
from aeon.datasets import load_from_tsv_file
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from utils import get_dataset_path, load_p2s_dataset, vae_encoding, get_model_and_hyperparams
from tqdm import tqdm
from typing import Callable
import pandas as pd
import numpy as np


# todo how to compare SAX and VAE if embeddings have different lengths?
#   SAX: n_segments, VAE: #patches * #symbols_per_patch -> dataset specific

def decision_tree_classifier(X_train, y_train, X_test, y_test):
    # Fit decision tree classifier
    clf = DecisionTreeClassifier().fit(X_train, y_train)

    # Calculate F1 and accuracy score
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    p = precision_score(y_test, y_pred, average='macro', zero_division=0)
    r = recall_score(y_test, y_pred, average='macro', zero_division=0)

    # print(f"Confusion matrix:\n{cm}")
    # print(f"F1 score: {f1:.2f}")
    # print(f"Accuracy: {acc:.2f}")

    return {"f1": f1, "accuracy": acc, "precision": p, "recall": r}


def get_sax_encoding(X, sax_params):
    sax = SAX(**sax_params)
    X_sax = sax.fit_transform(X).squeeze()
    return X_sax


def get_vae_encoding(X, model_name):
    vae, params = get_model_and_hyperparams(model_name)
    X_vae = vae_encoding(vae, X, params.patch_len)
    return X_vae


def run_experiment(dataset: str, iters_per_setting: int, enc_function: Callable, model_name: str, params):
    # Load datasets
    if dataset == "p2s":
        X_train, y_train = load_p2s_dataset("train")
        X_test, y_test = load_p2s_dataset("test")
    else:
        X_train, y_train = load_from_tsv_file(get_dataset_path(dataset, "train"))
        X_test, y_test = load_from_tsv_file(get_dataset_path(dataset, "test"))

    # todo adjust to models
    scaler = StandardScaler()
    X_train = np.expand_dims(scaler.fit_transform(X_train.squeeze()), axis=1)
    X_test = np.expand_dims(scaler.transform(X_test.squeeze()), axis=1)

    # Get encodings
    X_train_emb = enc_function(X_train, params)
    X_test_emb = enc_function(X_test, params)

    # Train decision tree classifier using encodings
    results = []
    for i in range(iters_per_setting):
        results.append(decision_tree_classifier(X_train_emb, y_train, X_test_emb, y_test))

    results = [d | {"dataset": dataset, "model": model_name} for d in results]
    return results


def classification(datasets, vae_models, sax_params, iters_per_setting):
    # Collect results across experiments
    all_results = []

    total_iters = len(datasets) * (len(vae_models) + 1)
    with tqdm(total=total_iters) as pbar:
        for dataset in datasets:
            for idx, sax_param in enumerate(sax_params):
                # SAX experiments
                sax_results = run_experiment(dataset, iters_per_setting, get_sax_encoding, f"sax_{idx}",
                                             params=sax_param)
                all_results = all_results + sax_results

                pbar.update(1)

            # VAE experiments
            for vae_model in vae_models:
                vae_results = run_experiment(dataset, iters_per_setting, get_vae_encoding, vae_model, params=vae_model) # todo a bit ugly?
                all_results = all_results + vae_results

                pbar.update(1)

    all_results = pd.DataFrame.from_records(all_results)
    all_results.to_csv("classification_results.csv")

    summary_df = pd.DataFrame()
    for metric in ["accuracy", "f1", "precision", "recall"]:
        res = all_results.groupby(["dataset", "model"], as_index=False)[metric].mean()
        if summary_df.empty:
            summary_df = res
        else:
            summary_df = summary_df.merge(res, on=["dataset", "model"], how="outer")

    print(summary_df)
    summary_df.to_csv(f"classification_results_summary.csv")


if __name__ == "__main__":
    # Define experiment parameters
    dataset_list = ["Wine", "Rock", "Plane", "ArrowHead", "p2s"]
    vae_list = ["fc", "model_xs", "xxxs1"]
    sax_list = [{"n_segments": 128, "alphabet_size": 8}]
    n_iters = 10

    classification(dataset_list, vae_list, sax_list, n_iters)