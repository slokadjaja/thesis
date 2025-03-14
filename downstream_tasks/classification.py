"""Compare performance of TS classification using SAX and VAE encodings"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import (
    get_or_create_experiment,
    get_dataset,
    get_sax_encoding,
    get_vae_encoding,
    get_vqshape_encoding,
)
from tqdm import tqdm
from typing import Callable
import pandas as pd
import numpy as np
import itertools
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
import os
import mlflow


class LinearClassifier(nn.Module):
    """
    A linear classifier with dropout on input features.
    """

    def __init__(self, input_dim, output_dim, dropout=0):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc(self.dropout(x))


def linear_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    num_epoch=100,
    weight_decay=0,
    dropout=0,
    batch_size=8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    train_loader = DataLoader(
        list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False
    )

    def train_and_validate(dropout, weight_decay):
        classifier = LinearClassifier(
            input_dim=X_train.shape[1],
            output_dim=len(np.unique(y_train)),
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.Adam(
            classifier.parameters(), lr=0.005, weight_decay=weight_decay
        )

        best_val_accuracy = 0
        for epoch in range(num_epoch):
            # Train
            classifier.train()
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = classifier(batch_features.to(device))
                loss = torch.nn.functional.cross_entropy(
                    outputs, batch_labels.view(-1).to(device)
                )
                loss.backward()
                optimizer.step()

            # Validation
            classifier.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    outputs = classifier(batch_features.to(device))
                    _, predicted = outputs.max(1)
                    total += batch_labels.size(0)
                    correct += (
                        predicted.eq(batch_labels.view(-1).to(device)).sum().item()
                    )

            val_accuracy = correct / total
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

        return best_val_accuracy, classifier

    # Do a bit of hyperparam search
    l2_list = [0, 0.01, 0.1, 1, 10, 100]
    dropout_list = [0, 0.2, 0.5, 0.8]
    hyp_combinations = list(itertools.product(l2_list, dropout_list))

    validation_accuracies = []
    classifiers = []
    print("Performing hyperparameter search (step 1/3)")
    for weight_decay, dropout in hyp_combinations:
        acc, clf = train_and_validate(dropout=dropout, weight_decay=weight_decay)
        validation_accuracies.append(acc)
        classifiers.append(clf)

    best_idx = np.argmax(validation_accuracies)
    best_classifier = classifiers[best_idx]

    # Training Accuracy
    best_classifier.eval()
    train_correct = 0
    train_total = 0
    print("Calculating training set accuracy (step 2/3)")
    with torch.no_grad():
        for batch_features, batch_labels in train_loader:
            outputs = best_classifier(batch_features.to(device))
            _, predicted = outputs.max(1)
            train_total += batch_labels.size(0)
            train_correct += predicted.eq(batch_labels.view(-1).to(device)).sum().item()

    train_acc = train_correct / train_total

    # Final Evaluation on Test Set
    best_classifier.eval()
    y_true = []
    y_pred = []
    print("Calculating metrics on test set (step 3/3)")
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            outputs = best_classifier(batch_features.to(device))
            _, predicted = outputs.max(1)
            y_true.extend(batch_labels.view(-1).cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Metrics Calculation
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    p = precision_score(y_true, y_pred, average="weighted")
    r = recall_score(y_true, y_pred, average="weighted")

    return {
        "train_accuracy": train_acc,
        "accuracy": acc,
        "f1": f1,
        "precision": p,
        "recall": r,
    }


def decision_tree_classifier(X_train, y_train, X_test, y_test):
    # Fit decision tree classifier
    clf = DecisionTreeClassifier().fit(X_train, y_train)

    # Calculate F1 and accuracy score
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    p = precision_score(y_test, y_pred, average="macro", zero_division=0)
    r = recall_score(y_test, y_pred, average="macro", zero_division=0)

    y_pred_train = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)

    # print(f"Confusion matrix:\n{cm}")
    # print(f"F1 score: {f1:.2f}")
    # print(f"Accuracy: {acc:.2f}")

    return {
        "train_accuracy": train_acc,
        "accuracy": acc,
        "f1": f1,
        "precision": p,
        "recall": r,
    }


def run_experiment(
    dataset: str,
    iters_per_setting: int,
    enc_function: Callable,
    model_name: str,
    params,
):
    X_train, y_train, X_test, y_test = get_dataset(dataset)

    # todo adjust to models
    scaler = StandardScaler()
    X_train = np.expand_dims(scaler.fit_transform(X_train), axis=1)
    X_test = np.expand_dims(scaler.transform(X_test), axis=1)

    # Get encodings
    X_train_emb = enc_function(X_train, params)
    X_test_emb = enc_function(X_test, params)

    # Train decision tree classifier using encodings
    results = []
    for i in range(iters_per_setting):
        results.append(
            decision_tree_classifier(X_train_emb, y_train, X_test_emb, y_test)
        )

    results = [d | {"dataset": dataset, "model": model_name} for d in results]
    return results


def classification(
    datasets: list[str],
    vae_models: list[str],
    sax_params: list[dict[str, int]],
    iters_per_setting: int,
    azure=True,
) -> None:
    """
    Perform time series classification using SAX and VAE-based encodings.

    This method evaluates the performance of time series classification using symbolic representations
    generated by SAX and VAEs. It trains  a Decision Tree classifier on these representations and computes metrics such
    as accuracy, F1-score, precision, and recall for each dataset and encoding method.

    Args:
        datasets: List of datasets to classify.
        vae_models: List of trained VAE model names to use for encoding.
        sax_params: List of dictionaries containing SAX parameters (n_segments and alphabet_size)
        iters_per_setting: Number of iterations to repeat the experiment for each dataset and encoding method.
        azure: Whether to use Azure ML for storing results. Defaults to True.
    """

    # Collect results across experiments
    all_results = []

    total_iters = len(datasets) * (len(vae_models) + len(sax_params) + 1)
    with tqdm(total=total_iters) as pbar:
        for dataset in datasets:
            # VQShape experiment
            vqshape_result = run_experiment(
                dataset, iters_per_setting, get_vqshape_encoding, "vqshape", None
            )
            all_results = all_results + vqshape_result
            pbar.update(1)

            # SAX experiments
            for idx, sax_param in enumerate(sax_params):
                sax_results = run_experiment(
                    dataset,
                    iters_per_setting,
                    get_sax_encoding,
                    f"sax_{idx}",
                    params=sax_param,
                )
                all_results = all_results + sax_results
                pbar.update(1)

            # VAE experiments
            for vae_model in vae_models:
                vae_results = run_experiment(
                    dataset,
                    iters_per_setting,
                    get_vae_encoding,
                    vae_model,
                    params=vae_model,
                )
                all_results = all_results + vae_results
                pbar.update(1)

    all_results = pd.DataFrame.from_records(all_results)
    all_results.to_csv("classification_results.csv")

    summary_df = pd.DataFrame()
    for metric in ["train_accuracy", "accuracy", "f1", "precision", "recall"]:
        res = all_results.groupby(["dataset", "model"], as_index=False)[metric].mean()
        if summary_df.empty:
            summary_df = res
        else:
            summary_df = summary_df.merge(res, on=["dataset", "model"], how="outer")

    print(summary_df)
    summary_df.to_csv("classification_results_summary.csv")

    if azure:
        load_dotenv()
        credential = ClientSecretCredential(
            os.environ["AZURE_TENANT_ID"],
            os.environ["AZURE_CLIENT_ID"],
            os.environ["AZURE_CLIENT_SECRET"],
        )
        mlflow.set_tracking_uri(os.environ["TRACKING_URI"])

        experiment_id = get_or_create_experiment("classification")
        mlflow.set_experiment(experiment_id=experiment_id)

        with mlflow.start_run(experiment_id=experiment_id, run_name="classification"):
            mlflow.log_artifact("classification_results.csv")
            mlflow.log_artifact("classification_results_summary.csv")


if __name__ == "__main__":
    # Define experiment parameters
    cls_datasets = ["Wine", "Rock", "Plane", "ArrowHead", "p2s", "FordA", "FordB"]
    vae_list = ["ArrowHead_p16_a32", "Wine_p16_a32", "p2s_p128_a32"]
    sax_params = [
        {"n_segments": 16, "alphabet_size": 32},
        {"n_segments": 16, "alphabet_size": 48},
    ]
    n_iters = 10

    classification(cls_datasets, vae_list, sax_params, n_iters, azure=False)
