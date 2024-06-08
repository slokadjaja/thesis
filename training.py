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

    parser.add_argument("--dataset", default="ArrowHead")
    parser.add_argument("--epoch", default=100)
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--lr", default=1e-4)
    parser.add_argument("--beta", default=1)
    # todo later also include parameters specified in model class

    return parser.parse_args()


if __name__ == "__main__":
    # set_seed(42)
    args = parse_args()

    # Define constants
    dataset = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = get_ts_length(dataset)
    num_epochs = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    beta = args.beta

    # Load dataset
    train = UCRDataset(dataset, "train")
    test = UCRDataset(dataset, "test")
    # shape of batch: [batch_size, 1, length]
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)

    # Define and train model
    model = VAE(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_arr = []
    loss = None

    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        for x, y in train_dataloader:
            x = x.to(device)

            logits, output = model(x)
            loss = reconstruction_loss(torch.squeeze(x, dim=1), output) + beta * cat_kl_div(logits, 10, 2)
            loss.backward()
            optimizer.step()

        loss_arr.append(loss.item())

    # Plot batch
    # for i in range(4):
    #     plt.plot(x[i].squeeze())

    # Plot loss
    fig, ax = plt.subplots()
    ax.plot(list(range(num_epochs)), loss_arr)
    ax.set(xlabel='Epoch', ylabel='Loss')

    plt.show()
