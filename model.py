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

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.alphabet_size*self.n_latent),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.alphabet_size*self.n_latent, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.input_dim),
            # nn.ReLU()
        )

    def forward(self, x):
        logits = self.encoder(x).view(-1, self.n_latent, self.alphabet_size)
        z = gumbel_softmax(logits, self.temperature)
        output = self.decoder(z)
        return logits, output

    def encode(self, x):
        logits = self.encoder(x).view(-1, self.n_latent, self.alphabet_size)
        z = gumbel_softmax(logits, 1e-3)    # set temp close to zero to turn softmax into argmax
        return torch.argmax(z, dim=-1)
