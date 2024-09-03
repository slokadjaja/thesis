import torch
from torch.utils.data import DataLoader
from dataset import UCRDataset
from utils import get_ts_length, cat_kl_div, reconstruction_loss, set_seed, Params
from model import VAE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import mlflow

if __name__ == "__main__":
    params = Params("params.json")

    # Define constants
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, num_epochs, batch_size, lr, beta, patch_len, normalize, norm_method, n_latent, alphabet_size,\
        temperature, arch, seed = params.dataset, params.epoch, params.batch_size, params.lr, params.beta, \
        params.patch_len, params.normalize, params.norm_method, params.n_latent, params.alphabet_size, \
        params.temperature, params.arch, params.seed

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
    loss_arr, rec_arr, kl_arr = [], [], []
    x, loss, rec_loss, kl_div = None, None, None, None

    exp = mlflow.set_experiment(experiment_id=arch)

    with mlflow.start_run():
        # Training loop
        vae.train()
        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            for x, y in train_dataloader:
                x = x.to(device)

                logits, output = vae(x)
                rec_loss = reconstruction_loss(torch.squeeze(x, dim=1), torch.squeeze(output, dim=1))
                kl_div = cat_kl_div(logits, n_latent=n_latent, alphabet_size=alphabet_size)
                loss = rec_loss + beta * kl_div

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_arr.append(loss.item())
            rec_arr.append(rec_loss.item())
            kl_arr.append(kl_div.item())

            mlflow.log_metrics({"total loss": loss.item(),
                                "reconstruction loss": rec_loss.item(),
                                "kl divergence": kl_div.item()}, step=epoch
                               )

        # Log hyperparams, model and model summary
        mlflow.log_params(params.dict)
        with open("model_summary.txt", "w") as f:
            f.write(repr(vae))
        mlflow.log_artifact("model_summary.txt")
        mlflow.pytorch.log_model(vae, "model")

        vae.eval()

        # Encode samples from training set
        # n = 500
        # with open('test_patch.npy', 'wb') as f:
        #     np.save(f, train[:n][0].detach().numpy().squeeze())
        #     np.save(f, vae.encode(train[:n][0]).detach().numpy().squeeze())

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
            ax.legend()
            ax.set(xlabel='Epoch', ylabel='Loss')
            plt.title(f"Loss Curve ({dataset})")
            # plt.savefig(f"loss_patch.png", dpi=300)
            plt.show()
