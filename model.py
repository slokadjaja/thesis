import torch
from torch import nn


def sample_gumbel(shape: torch.Size, device: torch.device, eps=1e-20) -> torch.Tensor:
    """Samples from the Gumbel distribution given a tensor shape and value of epsilon for numerical stability"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(
    logits: torch.Tensor, temperature: float, device: torch.device
) -> torch.Tensor:
    """Adds Gumbel noise to `logits` and applies softmax along the last dimension"""
    input_shape = logits.shape
    y = logits + sample_gumbel(logits.shape, device=device)
    return torch.nn.functional.softmax(y / temperature, dim=-1).view(input_shape)


def conv_out_len_multiple(l_in, arr):
    # arr: contain tuples (kernel, stride, pad)
    out_len_arr = [l_in]
    for a in arr:
        out_len_arr.append(conv_out_len(out_len_arr[-1], *a))

    return out_len_arr[-1]


def conv_out_len(l_in, kernel, stride, pad=0):
    import math

    return math.floor(1 + (l_in + 2 * pad - kernel) / stride)


class VAE(nn.Module):
    def __init__(
        self,
        input_dim,
        alphabet_size=2,
        n_latent=10,
        temperature=1,
        model="fc",
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.alphabet_size = alphabet_size
        self.n_latent = n_latent
        self.temperature = temperature
        self.model = model
        self.device = device

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
            nn.Linear(500, self.alphabet_size * self.n_latent),
        )

        fc_decoder = nn.Sequential(
            nn.Flatten(),
            nn.Unflatten(1, (1, self.alphabet_size * self.n_latent)),
            nn.Dropout(0.3),
            nn.Linear(self.alphabet_size * self.n_latent, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(500, self.input_dim),
        )

        len_after_conv = conv_out_len_multiple(
            self.input_dim, [(8, 1, 0), (5, 1, 0), (3, 1, 0)]
        )
        cnn_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(len_after_conv * 128, self.alphabet_size * self.n_latent),
        )

        cnn_decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.alphabet_size * self.n_latent, len_after_conv * 128),
            nn.Unflatten(dim=1, unflattened_size=(128, len_after_conv)),
            nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=5),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=1, kernel_size=8),
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
        z = gumbel_softmax(logits, self.temperature, self.device)
        output = self.decoder(z)
        return logits, output

    def encode(self, x):
        logits = self.encoder(x).view(-1, self.n_latent, self.alphabet_size)
        # try argmax instead of sampling
        # z = gumbel_softmax(logits, 1e-3)  # set temp close to zero to turn softmax into argmax
        return torch.argmax(logits, dim=-1)
