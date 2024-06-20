import torch
from torch import nn
from utils import gumbel_softmax


class VAE(nn.Module):
    def __init__(self, input_dim, h_dim=100, alphabet_size=2, n_latent=10, temperature=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.alphabet_size = alphabet_size
        self.n_latent = n_latent
        self.temperature = temperature

        # LazyLinear layer could be helpful
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.alphabet_size * self.n_latent),
            nn.ReLU()
        )

        self.fc_decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.alphabet_size * self.n_latent, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.input_dim),
            # nn.ReLU()
        )

        # TODO: CNN only works for ArrowHead + full ts, have to adjust dimensions for other datasets
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=4, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=4),
            nn.Flatten(),
            nn.Linear(in_features=62, out_features=32),
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=self.alphabet_size * self.n_latent),
            nn.LeakyReLU(),
        )

        self.cnn_decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.alphabet_size * self.n_latent, out_features=32),
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=62),
            nn.LeakyReLU(),
            nn.Unflatten(dim=1, unflattened_size=(2, 31)),
            # 2: channels, 31: length after maxpool in encoder, before flatten
            nn.Upsample(size=123),  # length before maxpool
            nn.ConvTranspose1d(in_channels=2, out_channels=2, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=2, out_channels=1, kernel_size=4, stride=1),
        )

    def forward(self, x):
        logits = self.cnn_encoder(x).view(-1, self.n_latent, self.alphabet_size)
        z = gumbel_softmax(logits, self.temperature)
        output = self.cnn_decoder(z)
        return logits, output

    def encode(self, x):
        logits = self.cnn_encoder(x).view(-1, self.n_latent, self.alphabet_size)
        z = gumbel_softmax(logits, 1e-3)  # set temp close to zero to turn softmax into argmax
        return torch.argmax(z, dim=-1)
