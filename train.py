import torch
from torch.utils.data import DataLoader
from dataset import UCRDataset
from utils import get_ts_length, cat_kl_div, reconstruction_loss, set_seed, Params, contrastive_loss
from model import VAE
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow
import os
from pathlib import Path
import shutil


if __name__ == "__main__":
    params = Params("params.json")

    # Define constants
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, num_epochs, batch_size, lr, beta, patch_len, normalize, norm_method, n_latent, alphabet_size, \
        temperature, arch, seed, top_quantile, bottom_quantile, margin, alpha, run_name = params.dataset, params.epoch, \
        params.batch_size, params.lr, params.beta, params.patch_len, params.normalize, params.norm_method, \
        params.n_latent, params.alphabet_size, params.temperature, params.arch, params.seed, params.top_quantile, \
        params.bottom_quantile, params.margin, params.alpha, params.run_name

    if seed:
        set_seed(seed)

    ts_length = get_ts_length(dataset)

    if patch_len is None:
        input_dim = ts_length
    else:
        input_dim = patch_len

    # Load dataset
    train = UCRDataset(dataset, "train", patch_len=patch_len, normalize=normalize, norm_method=norm_method)
    # shape of batch: [batch_size, 1, length]
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)

    # Define and train model
    vae = VAE(input_dim, alphabet_size, n_latent, temperature, arch).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # Track entire loss, and also components (reconstruction loss, kl divergence)
    loss_arr, rec_arr, kl_arr, closs_arr = [], [], [], []
    x, loss, rec_loss, kl_div, closs = None, None, None, None, None

    # exp = mlflow.set_experiment(experiment_id=arch)

    with mlflow.start_run(run_name=run_name) as run:
        # Training loop
        vae.train()
        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            for x, y in train_dataloader:
                x = x.to(device)

                logits, output = vae(x)
                rec_loss = reconstruction_loss(torch.squeeze(x, dim=1), torch.squeeze(output, dim=1))
                kl_div = cat_kl_div(logits, n_latent=n_latent, alphabet_size=alphabet_size)
                closs = contrastive_loss(x, top_quantile, bottom_quantile, margin)

                loss = rec_loss + beta * kl_div + alpha * closs

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

        # MLFlow: Log model and params
        mlflow.pytorch.log_model(vae, "model")
        torch.save(vae.state_dict(), "model.pt")
        mlflow.log_artifact("model.pt")

        # Save model.pt and params.json to baseline_models folder
        Path(f"baseline_models/{run_name}").mkdir(parents=True, exist_ok=True)
        shutil.copy("model_summary.txt", f"baseline_models/{run_name}/model_summary.txt")
        shutil.copy("model.pt", f"baseline_models/{run_name}/model.pt")
        shutil.copy("params.json", f"baseline_models/{run_name}/params.json")

        os.remove("model_summary.txt")
        os.remove("model.pt")
        
        vae.eval()

        # Plot reconstruction example
        with torch.no_grad():
            for i in range(10):  # min(batch_size, len(x))):
                # input shape should be [1, 1, * ] instead of [1, * ], otherwise flatten in cnn_encoder does not work
                sample = torch.unsqueeze(x[i], 0)
                fig, ax = plt.subplots()
                ax.plot(sample.squeeze(), label="ground truth")
                ax.plot(vae(sample)[1].detach().numpy().squeeze(), label="reconstruction")
                ax.legend()
                plt.title(f"Reconstruction Example ({dataset})")
                mlflow.log_figure(fig, f'recon_{i}.png')
                # plt.savefig(f"recon_patch_{idx}.png", dpi=300)
                plt.show()

            # Plot loss
            fig, ax = plt.subplots()
            ax.plot(loss_arr, label="Loss")
            ax.plot(kl_arr, alpha=0.6, label="KL divergence")
            ax.plot(rec_arr, alpha=0.6, label="Reconstruction loss")
            ax.plot(closs_arr, alpha=0.6, label="Contrastive loss")
            ax.legend()
            ax.set(xlabel='Epoch', ylabel='Loss')
            plt.title(f"Loss Curve ({dataset})")
            # plt.savefig(f"loss_patch.png", dpi=300)
            plt.show()
