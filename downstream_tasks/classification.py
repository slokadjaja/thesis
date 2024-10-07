"""Compare performance of TS classification using SAX and VAE encodings"""

from aeon.transformations.collection.dictionary_based import SAX
from aeon.datasets import load_from_tsv_file
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from utils import *
from model import VAE
import torch
from tqdm import tqdm
from typing import Callable
import pandas as pd


# todo how to compare SAX and VAE if embeddings have different lengths?
#   SAX: n_segments, VAE: #patches * #symbols_per_patch -> dataset specific

def decision_tree_classifier(X_train, y_train, X_test, y_test):
    # Fit decision tree classifier
    clf = DecisionTreeClassifier().fit(X_train, y_train)

    # Calculate F1 and accuracy score
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # print(f"Confusion matrix:\n{cm}")
    # print(f"F1 score: {f1:.2f}")
    # print(f"Accuracy: {acc:.2f}")

    return {"f1": f1, "accuracy": acc}


def get_sax_encoding(X, sax_params):
    sax = SAX(**sax_params)
    X_sax = sax.fit_transform(X).squeeze()
    return X_sax


def get_vae_encoding(X, vae_params):
    params = Params(vae_params["params_path"])
    vae = VAE(params.patch_len, params.alphabet_size, params.n_latent, params.arch)
    vae.load_state_dict(torch.load(vae_params["model_path"]))
    vae.eval()
    X_vae = vae_encoding(vae, X, params.patch_len)
    return X_vae


def run_experiment(dataset: str, iters_per_setting: int, enc_function: Callable, params: dict[str, str | int],
                   model_name: str):
    # Load datasets
    X_train, y_train = load_from_tsv_file(get_dataset_path(dataset, "train"))
    X_test, y_test = load_from_tsv_file(get_dataset_path(dataset, "test"))

    # Get encodings
    X_train_emb = enc_function(X_train, params)
    X_test_emb = enc_function(X_test, params)

    # Train decision tree classifier using encodings
    results = []
    for i in range(iters_per_setting):
        results.append(decision_tree_classifier(X_train_emb, y_train, X_test_emb, y_test))

    results = [d | {"dataset": dataset, "model": model_name} for d in results]
    return results


def main():
    # Define experiment parameters
    datasets = ["Wine", "Rock", "Plane", "ArrowHead"]
    vae_models = [
        {"model_path": "../baseline_models/fc/model.pt",
         "params_path": "../baseline_models/fc/params.json"},
        {"model_path": "../baseline_models/model_contrastive/model.pt",
         "params_path": "../baseline_models/model_contrastive/params.json"},
        {"model_path": "../baseline_models/model_s/model.pt",
         "params_path": "../baseline_models/model_s/params.json"},
        {"model_path": "../baseline_models/model_xs/model.pt",
         "params_path": "../baseline_models/model_xs/params.json"}
    ]
    sax_params = {"n_segments": 128, "alphabet_size": 8}
    iters_per_setting = 10

    # Collect results across experiments
    all_results = []

    total_iters = len(datasets) * len(vae_models)
    with tqdm(total=total_iters) as pbar:
        for dataset in datasets:
            # SAX experiments
            sax_results = run_experiment(dataset, iters_per_setting, get_sax_encoding, sax_params, "sax")
            all_results = all_results + sax_results

            # VAE experiments
            for vae_model in vae_models:
                model_name = vae_model['model_path'].split('/')[-2]
                vae_results = run_experiment(dataset, iters_per_setting, get_vae_encoding, vae_model, model_name)
                all_results = all_results + vae_results

                pbar.update(1)

    all_results = pd.DataFrame.from_records(all_results)
    all_results.to_csv("exp_results.csv")

    for metric in ["accuracy", "f1"]:
        res = all_results.groupby(["dataset", "model"], as_index=False)[metric].mean()
        res.to_csv(f"exp_results_{metric}.csv")
        print(metric)
        print(res)


if __name__ == "__main__":
    main()
