import torch
from torch.utils.data import DataLoader
from dataset import TSDataset
from utils import get_ts_length, cat_kl_div, reconstruction_loss, set_seed, Params, triplet_loss, \
    plot_reconstructions, plot_loss, get_or_create_experiment
from model import VAE
from tqdm import tqdm
import mlflow
from pathlib import Path
import json
import os

from dotenv import load_dotenv
from azure.identity import ClientSecretCredential


class Trainer:
    def __init__(self, params, experiment_name="train", run_name="test_run", azure=True):
        if azure:
            load_dotenv()
            credential = ClientSecretCredential(os.environ["AZURE_TENANT_ID"], os.environ["AZURE_CLIENT_ID"],
                                                os.environ["AZURE_CLIENT_SECRET"])
            mlflow.set_tracking_uri(os.environ["TRACKING_URI"])

        self.experiment_name = experiment_name
        self.run_name = run_name

        self.params = params
        self.seed = params.seed

        if self.seed:
            set_seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = get_ts_length(params.dataset) if params.patch_len is None else params.patch_len
        self.model = VAE(self.input_dim, params.alphabet_size, params.n_latent, params.temperature, params.arch,
                         self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)

        # Prepare dataloader
        train_dataset = TSDataset(params.dataset, "train", patch_len=params.patch_len, normalize=params.normalize,
                                   norm_method=params.norm_method)
        self.train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

        # Initialize metrics lists
        self.loss_arr, self.rec_arr, self.kl_arr, self.closs_arr = [], [], [], []

        # Directory name to save results
        self.dir_name = "models"

    def train_one_epoch(self):
        self.model.train()
        epoch_loss, epoch_rec_loss, epoch_kl_div, epoch_closs = 0.0, 0.0, 0.0, 0.0
        for x, y in self.train_dataloader:
            x = x.to(self.device)  # shape of batch: [batch_size, 1, length]

            # Calculate loss
            logits, output = self.model(x)
            rec_loss = reconstruction_loss(torch.squeeze(x, dim=1), torch.squeeze(output, dim=1))
            kl_div = cat_kl_div(logits, n_latent=self.params.n_latent, alphabet_size=self.params.alphabet_size)
            closs = triplet_loss(x, logits, self.params.top_quantile, self.params.bottom_quantile, self.params.margin)
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

        with mlflow.start_run(experiment_id=experiment_id, run_name=self.run_name):
            mlflow.log_params(self.params.dict)
            for epoch in tqdm(range(self.params.epoch), desc="Epoch"):
                loss, rec_loss, kl_div, closs = self.train_one_epoch()
                mlflow.log_metrics({"total loss": loss, "reconstruction loss": rec_loss, "kl divergence": kl_div,
                                    "contrastive loss": closs}, step=epoch)

            self.save_model_artifacts()
            self.plot_results()

    def save_model_artifacts(self):
        """Save model and architecture details"""
        model_dir = Path(f"{self.dir_name}/{self.run_name}")
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
        plot_dir = Path(f"{self.dir_name}/{self.run_name}/plots")
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
                mlflow.log_figure(fig, f'recon_{i}.png')

        # Plot loss
        loss_fig = plot_loss({
            "Total loss": self.loss_arr,
            "KL divergence": self.kl_arr,
            "Reconstruction loss": self.rec_arr,
            "Contrastive loss": self.closs_arr
        }, self.params.dataset)
        loss_fig_path = plot_dir / "loss.png"
        loss_fig.savefig(loss_fig_path, dpi=300)


if __name__ == "__main__":
    params = Params("params.json")
    trainer = Trainer(params)
    trainer.train()
