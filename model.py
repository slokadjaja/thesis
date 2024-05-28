import torch
from torch import nn
from utils import gumbel_softmax


class VAE(nn.Module):
    def __init__(self, input_dim, h_dim=100, n_classes=2, n_latent=10, temperature=0.1):
        super().__init__()

        self.n_classes = n_classes
        self.n_latent = n_latent
        self.temperature = temperature

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, n_classes*n_latent),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_classes*n_latent, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, input_dim),
            # nn.ReLU()
        )

    def forward(self, x):
        logits = self.encoder(x).view(-1, self.n_latent, self.n_classes)
        z = gumbel_softmax(logits, self.temperature)
        output = self.decoder(z)
        return logits, output

    def encode(self, x):
        logits = self.encoder(x).view(-1, self.n_latent, self.n_classes)
        z = gumbel_softmax(logits, 1e-3)    # set temp close to zero to turn softmax into argmax
        return torch.argmax(z, dim=-1)
