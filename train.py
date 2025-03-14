import torch
from torch.utils.data import DataLoader
import torch.distributions as dist
import torch.nn.functional as F
from dataset import TSDataset
from utils import (
    get_ts_length,
    set_seed,
    Params,
    plot_reconstructions,
    plot_loss,
    get_or_create_experiment,
)
from model import VAE
from tqdm import tqdm
from tslearn.metrics import dtw
from itertools import combinations
import numpy as np
import mlflow
from pathlib import Path
import json
import os
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential


def cat_kl_div(logits, n_latent, alphabet_size):
    q = dist.Categorical(logits=logits)
    p = dist.Categorical(
        probs=torch.full(
            (n_latent, alphabet_size), 1.0 / alphabet_size, device=logits.device
        )
    )
    kl = dist.kl.kl_divergence(q, p)
    return torch.mean(torch.sum(kl, dim=1))


def reconstruction_loss(x_true, x_out):
    return F.mse_loss(x_true, x_out)


def compute_thresholds(patches, method="dtw", lower=25, upper=75):
    """Compute distance-based thresholds for positive and negative selection."""
    distances = []

    comb = list(combinations(range(len(patches)), 2))
    total_iters = len(comb)

    with tqdm(total=total_iters, desc="Combinations: ") as pbar:
        for i, j in combinations(range(len(patches)), 2):  # Pairwise distances
            if method == "dtw":
                dist = dtw(
                    patches[i].squeeze().cpu().detach().numpy(),
                    patches[j].squeeze().cpu().detach().numpy(),
                )
            elif method == "l2":
                dist = np.linalg.norm(
                    patches[i].squeeze().cpu().detach().numpy()
                    - patches[j].squeeze().cpu().detach().numpy()
                )
            distances.append(dist)
            pbar.update(1)

    # Convert to numpy for easier percentile calculations
    distances = np.array(distances)

    # Set thresholds based on distribution percentiles
    T_pos = np.percentile(distances, lower)
    T_neg = np.percentile(distances, upper)
    T_semi = np.median(distances)  # Middle ground for semi-hard negatives

    return T_pos, T_neg, T_semi


def compute_dtw_distance(data):
    """Computes pairwise DTW distances for a batch of time series."""
    batch_len = data.shape[0]
    pairwise_dtw = torch.zeros(
        (batch_len, batch_len), device=data.device
    )  # Store DTW distances

    for i in range(batch_len):
        for j in range(i + 1, batch_len):  # Compute only upper triangle
            dtw_dist = dtw(
                data[i].cpu().detach().numpy(), data[j].cpu().detach().numpy()
            )
            pairwise_dtw[i, j] = dtw_dist
            pairwise_dtw[j, i] = dtw_dist  # Symmetric matrix

    return pairwise_dtw


def triplet_loss(
    batch, labels, logits, neg_threshold, pos_threshold, m, dist_metric="dtw"
):
    """Computes the triplet loss.

    For each sample in a batch, look for positive and negative samples, then calculate the triplet loss using their
    embeddings.

    Args:
        batch: A list of time series patches, shape: [batch_size, 1, patch_length]
        labels: Class labels belonging to the patches
        logits: Logits tensor of shape [batch_size, n_latent, alphabet_size]
        neg_threshold: distance threshold for a negative pair
        pos_threshold: distance threshold for a positive pair
        m: margin for triplet loss
        dist_metric: distance metric to use for patches
    Returns:
        Average triplet loss over found triplets.
    """

    data = torch.squeeze(batch)
    batch_len = data.shape[0]
    if dist_metric == "l2":
        pair_distance = torch.cdist(data, data, p=2)
    elif dist_metric == "dtw":
        pair_distance = compute_dtw_distance(data)
    else:
        raise ValueError("Metric not available")

    triplet_loss = torch.tensor(0, device=batch.device)
    num_triplets = 0

    for i in range(batch_len):
        anchor = logits[i]

        # Sample positive and negative patches

        # Select positive indices (same class, small distance, excluding self)
        pos_indices = torch.where(
            (labels == labels[i])
            & (pair_distance[i] <= pos_threshold)
            & (torch.arange(len(data), device=pair_distance.device) != i)
        )[0]
        # Select positive indices (same class, excluding self)
        # pos_indices = torch.where((labels == labels[i]) & (torch.arange(batch_len, device=labels.device) != i))[0]
        if torch.numel(pos_indices) != 0:
            pos_rand = torch.randint(
                0, len(pos_indices), size=(1,), device=batch.device
            ).item()
            pos_sample = logits[pos_indices[pos_rand]]
        else:
            continue

        # Select negative indices (different class, large distance)
        neg_indices = torch.where(
            (labels != labels[i]) & (pair_distance[i] >= neg_threshold)
        )[0]
        # Select negative indices (different class)
        # neg_indices = torch.where((labels != labels[i]))[0]
        if torch.numel(neg_indices) != 0:
            neg_rand = torch.randint(
                0, len(neg_indices), size=(1,), device=batch.device
            ).item()
            neg_sample = logits[neg_indices[neg_rand]]
        else:
            continue

        triplet_loss = triplet_loss + F.triplet_margin_loss(
            anchor=anchor, positive=pos_sample, negative=neg_sample, margin=m
        )
        num_triplets = num_triplets + 1

    return triplet_loss / (num_triplets + 1e-6)  # avoid division by zero


def temp_exp_annealing(initial_temp, epoch, decay_rate=0.99):
    """
    Exponentially decay temperature using a decay rate.
    """
    return initial_temp * (decay_rate**epoch)


class Trainer:
    def __init__(
        self,
        params,
        experiment_name="train",
        run_name="test_run",
        azure=True,
        component=None,
    ):
        if azure:
            load_dotenv()
            credential = ClientSecretCredential(
                os.environ["AZURE_TENANT_ID"],
                os.environ["AZURE_CLIENT_ID"],
                os.environ["AZURE_CLIENT_SECRET"],
            )
            mlflow.set_tracking_uri(os.environ["TRACKING_URI"])

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.component = component
        self.params = params
        self.seed = params.seed
        if self.seed:
            set_seed(self.seed)

        # if component is set, then this model is part of a larger run containing multiple models
        self.nested = True if component is not None else False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = (
            get_ts_length(params.dataset)
            if params.patch_len is None
            else params.patch_len
        )
        self.model = VAE(
            self.input_dim,
            params.alphabet_size,
            params.n_latent,
            params.temperature,
            params.arch,
            self.device,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)

        # Prepare dataloader
        train_dataset = TSDataset(
            params.dataset,
            "train",
            patch_len=params.patch_len,
            normalize=params.normalize,
            norm_method=params.norm_method,
            component=self.component,
        )
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=params.batch_size, shuffle=True
        )

        # Compute thresholds for triplet loss
        self.pos_threshold, self.neg_threshold, self.mid_threshold = compute_thresholds(
            train_dataset.x
        )

        # Initialize metrics lists
        self.loss_arr, self.rec_arr, self.kl_arr, self.closs_arr = [], [], [], []

        # Directory name to save results
        self.dir_name = "models"
        base_path = f"{self.dir_name}/{self.run_name}"
        self.model_path = (
            base_path if self.component is None else f"{base_path}/{self.component}"
        )

    def train_one_epoch(self):
        self.model.train()
        epoch_loss, epoch_rec_loss, epoch_kl_div, epoch_closs = 0.0, 0.0, 0.0, 0.0
        for x, y in self.train_dataloader:
            x = x.to(self.device)  # shape of batch: [batch_size, 1, length]

            # Calculate loss
            logits, output = self.model(x)
            rec_loss = reconstruction_loss(
                torch.squeeze(x, dim=1), torch.squeeze(output, dim=1)
            )
            kl_div = cat_kl_div(
                logits,
                n_latent=self.params.n_latent,
                alphabet_size=self.params.alphabet_size,
            )
            closs = triplet_loss(
                x, y, logits, self.neg_threshold, self.pos_threshold, self.params.margin
            )
            loss = rec_loss + self.params.beta * kl_div + self.params.alpha * closs

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_rec_loss += rec_loss.item()
            epoch_kl_div += kl_div.item()
            epoch_closs += closs.item()

        mean_loss = epoch_loss / len(self.train_dataloader)
        mean_rec_loss = epoch_rec_loss / len(self.train_dataloader)
        mean_kl_div = epoch_kl_div / len(self.train_dataloader)
        mean_closs = epoch_closs / len(self.train_dataloader)

        self.loss_arr.append(mean_loss)
        self.rec_arr.append(mean_rec_loss)
        self.kl_arr.append(mean_kl_div)
        self.closs_arr.append(mean_closs)

        return mean_loss, mean_rec_loss, mean_kl_div, mean_closs

    def train(self):
        experiment_id = get_or_create_experiment(self.experiment_name)
        mlflow.set_experiment(experiment_id=experiment_id)

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=self.run_name if self.component is None else self.component,
            nested=self.nested,
        ):
            mlflow.log_params(self.params.dict)
            for epoch in tqdm(range(self.params.epoch), desc="Epoch"):
                loss, rec_loss, kl_div, closs = self.train_one_epoch()
                if self.params.annealing:
                    current_temp = temp_exp_annealing(self.params.temperature, epoch)
                    self.model.temperature = current_temp

                mlflow.log_metrics(
                    {
                        "total loss": loss,
                        "reconstruction loss": rec_loss,
                        "kl divergence": kl_div,
                        "contrastive loss": closs,
                    },
                    step=epoch,
                )

            self.save_model_artifacts()
            self.plot_results()

    def save_model_artifacts(self):
        """Save model and architecture details"""
        model_dir = Path(self.model_path)
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), model_dir / "model.pt")
        with open(model_dir / "model_summary.txt", "w") as f:
            f.write(repr(self.model))
        with open(model_dir / "params.json", "w") as f:
            json.dump(self.params.dict, f)

        mlflow.log_artifact(str(model_dir / "model.pt"))
        mlflow.log_artifact(str(model_dir / "model_summary.txt"))
        mlflow.log_artifact(str(model_dir / "params.json"))

    def plot_results(self):
        """Plot reconstruction examples and training results"""
        plot_dir = Path(f"{self.model_path}/plots")
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Plot reconstruction examples
        if self.params.plot_recon:
            self.model.eval()
            with torch.no_grad():
                x, _ = next(iter(self.train_dataloader))
                figs = plot_reconstructions(self.model, x.to(self.device), 10)
            for i, fig in enumerate(figs):
                fig_path = plot_dir / f"recon_{i}.png"
                fig.savefig(fig_path, dpi=300)
                mlflow.log_figure(fig, f"recon_{i}.png")

        # Plot loss
        loss_fig = plot_loss(
            {
                "Total loss": self.loss_arr,
                "KL divergence": self.kl_arr,
                "Reconstruction loss": self.rec_arr,
                "Contrastive loss": self.closs_arr,
            },
            self.params.dataset,
        )
        loss_fig_path = plot_dir / "loss.png"
        loss_fig.savefig(loss_fig_path, dpi=300)


if __name__ == "__main__":
    params = Params("params.json")
    trainer = Trainer(params, azure=False, run_name="ArrowHead_p16_a32_t4")
    trainer.train()
