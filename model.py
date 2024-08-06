import torch
from torch import nn
from utils import gumbel_softmax, conv_out_len_multiple


class VAE(nn.Module):
    def __init__(self, input_dim, alphabet_size=2, n_latent=10, temperature=1, model="fc"):
        super().__init__()

        self.input_dim = input_dim
        self.alphabet_size = alphabet_size
        self.n_latent = n_latent
        self.temperature = temperature
        self.model = model

        fc_encoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.input_dim, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(500, self.alphabet_size*self.n_latent)
        )

        fc_decoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.alphabet_size*self.n_latent, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(500, self.input_dim)
        )

        cnn_encoder = nn.Sequential(
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

        cnn_decoder = nn.Sequential(

        )

        if self.model == "fc":
            self.encoder = fc_encoder
            self.decoder = fc_decoder
        elif self.model == "cnn":
            self.encoder = cnn_encoder
            self.decoder = cnn_decoder
        else:
            raise Exception("choose between fc and cnn")

    def forward(self, x):
        logits = self.encoder(x).view(-1, self.n_latent, self.alphabet_size)
        z = gumbel_softmax(logits, self.temperature)
        output = self.decoder(z)
        return logits, output

    def encode(self, x):
        logits = self.encoder(x).view(-1, self.n_latent, self.alphabet_size)
        z = gumbel_softmax(logits, 1e-3)  # set temp close to zero to turn softmax into argmax
        return torch.argmax(z, dim=-1)
