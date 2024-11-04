import torch
from torch.utils.data import DataLoader
from dataset import UCRDataset
from utils import get_ts_length, cat_kl_div, reconstruction_loss, set_seed, Params, contrastive_loss, \
    plot_reconstructions, plot_loss
from model import VAE
from tqdm import tqdm
import mlflow
import os
from pathlib import Path
import shutil


if __name__ == "__main__":
    params = Params("params.json")

    # Define constants
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if params.seed:
        set_seed(params.seed)

    ts_length = get_ts_length(params.dataset)

    if params.patch_len is None:
        input_dim = ts_length
    else:
        input_dim = params.patch_len

    # Load dataset
    train = UCRDataset(params.dataset, "train", patch_len=params.patch_len, normalize=params.normalize,
                       norm_method=params.norm_method)
    # shape of batch: [batch_size, 1, length]
    train_dataloader = DataLoader(train, batch_size=params.batch_size, shuffle=True)

    # Define and train model
    vae = VAE(input_dim, params.alphabet_size, params.n_latent, params.temperature, params.arch).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=params.lr)

    # Track entire loss, and also components (reconstruction loss, kl divergence)
    loss_arr, rec_arr, kl_arr, closs_arr = [], [], [], []
    x, loss, rec_loss, kl_div, closs = None, None, None, None, None

    # exp = mlflow.set_experiment(experiment_id=params.arch)

    with mlflow.start_run(run_name=params.run_name) as run:
        # Training loop
        vae.train()
        for epoch in tqdm(range(params.epoch), desc="Epoch"):
            for x, y in train_dataloader:
                x = x.to(device)

                logits, output = vae(x)
                rec_loss = reconstruction_loss(torch.squeeze(x, dim=1), torch.squeeze(output, dim=1))
                kl_div = cat_kl_div(logits, n_latent=params.n_latent, alphabet_size=params.alphabet_size)
                closs = contrastive_loss(x, params.top_quantile, params.bottom_quantile, params.margin)

                loss = rec_loss + params.beta * kl_div + params.alpha * closs

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_arr.append(loss.item())
            rec_arr.append(rec_loss.item())
            kl_arr.append(kl_div.item())
            closs_arr.append(closs.item())

            mlflow.log_metrics({
                "total loss": loss.item(), "reconstruction loss": rec_loss.item(), "kl divergence": kl_div.item(),
                "contrastive loss": closs.item()
                }, step=epoch
            )

        # MLFlow: Log hyperparams
        mlflow.log_params(params.dict)

        # MLFlow: Log model architecture
        with open("model_summary.txt", "w") as f:
            f.write(repr(vae))
        mlflow.log_artifact("model_summary.txt")

        # MLFlow: Log model params
        torch.save(vae.state_dict(), "model.pt")
        mlflow.log_artifact("model.pt")

        # Save model.pt and params.json to baseline_models folder
        Path(f"baseline_models/{params.run_name}").mkdir(parents=True, exist_ok=True)
        shutil.copy("model_summary.txt", f"baseline_models/{params.run_name}/model_summary.txt")
        shutil.copy("model.pt", f"baseline_models/{params.run_name}/model.pt")
        shutil.copy("params.json", f"baseline_models/{params.run_name}/params.json")

        os.remove("model_summary.txt")
        os.remove("model.pt")
        
        vae.eval()

        with torch.no_grad():
            # Plot reconstruction examples
            if params.plot_recon:
                figs = plot_reconstructions(vae, x, 10)
                for i, fig in enumerate(figs):
                    mlflow.log_figure(fig, f'recon_{i}.png')
                    fig.savefig(f'recon_{i}.png', dpi=300)

            # Plot loss
            loss_fig = plot_loss({"Total loss": loss_arr, "KL divergence": kl_arr,
                                  "Reconstruction loss": rec_arr, "Contrastive loss": closs_arr}, params.dataset)
            loss_fig.savefig(f"loss.png", dpi=300)