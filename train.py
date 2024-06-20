import torch
from torch.utils.data import DataLoader
from dataset import UCRDataset
from utils import get_ts_length, cat_kl_div, reconstruction_loss, set_seed
from model import VAE
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Training parameters')

    parser.add_argument("--dataset", type=str, default="ArrowHead")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=int, default=1)
    parser.add_argument("--patch_len", type=int, default=None)
    # todo later also include parameters specified in model class

    return parser.parse_args()


if __name__ == "__main__":
    # set_seed(42)
    args = parse_args()

    # Define constants
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, num_epochs, batch_size, lr, beta, patch_len = \
        args.dataset, args.epoch, args.batch_size, args.lr, args.beta, args.patch_len
    ts_length = get_ts_length(dataset)

    # Load dataset
    train = UCRDataset(dataset, "train", patch_len=patch_len)
    test = UCRDataset(dataset, "test", patch_len=patch_len)
    # shape of batch: [batch_size, 1, length]
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)

    if patch_len is None:
        input_dim = ts_length
    else:
        input_dim = patch_len

    n_latent = input_dim // 4
    alphabet_size = 2

    # Define and train model
    model = VAE(input_dim=input_dim,
                h_dim=input_dim // 2,
                n_latent=n_latent,
                alphabet_size=alphabet_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # track entire loss, and also components (reconstruction loss, kl divergence)
    loss_arr, rec_arr, kl_arr = [], [], []
    x, loss, rec_loss, kl_div = None, None, None, None

    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        for x, y in train_dataloader:
            x = x.to(device)

            logits, output = model(x)
            rec_loss = reconstruction_loss(torch.squeeze(x, dim=1), output)
            kl_div = cat_kl_div(logits, n_latent=n_latent, alphabet_size=alphabet_size)
            loss = rec_loss + beta * kl_div

            loss.backward()
            optimizer.step()

        loss_arr.append(loss.item())
        rec_arr.append(rec_loss.item())
        kl_arr.append(kl_div.item())

    # Encode samples from training set
    n = 500
    with open('test_patch.npy', 'wb') as f:
        np.save(f, train[:n][0].detach().numpy().squeeze())
        np.save(f, model.encode(train[:n][0]).detach().numpy().squeeze())

    # Plot reconstruction example
    idx = 0
    plt.plot(x[idx].squeeze(), label="ground truth")
    plt.plot(model(x[idx])[1].detach().numpy().squeeze(), label="reconstruction")
    plt.legend()
    plt.title(f"Reconstruction Example ({dataset})")
    # plt.savefig(f"recon_patch_{idx}.png", dpi=300)

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
