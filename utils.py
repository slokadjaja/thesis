from pathlib import Path
import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F


def get_dataset_path(name, split):
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir / "datasets_ucr"

    if split == "train":
        path = data_dir / f"{name}/{name}_TRAIN.tsv"
    elif split == "test":
        path = data_dir / f"{name}/{name}_TEST.tsv"
    else:
        raise Exception('Split should be either "train" or "test"')

    return path


def get_ts_length(name):
    arr = np.loadtxt(get_dataset_path(name, "train"), delimiter='\t')
    return len(arr[0]) - 1


def sample_gumbel(shape: torch.Size, eps=1e-20) -> torch.Tensor:
    """ Samples from the Gumbel distribution given a tensor shape and value of epsilon for numerical stability """
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """ Adds Gumbel noise to `logits` and applies softmax along the last dimension """
    input_shape = logits.shape
    y = logits + sample_gumbel(logits.shape)
    return torch.nn.functional.softmax(y / temperature, dim=-1).view(input_shape)


def cat_kl_div(logits, n_latent, n_classes):
    q = dist.Categorical(logits=logits)
    p = dist.Categorical(probs=torch.full((n_latent, n_classes), 1.0/n_classes))
    kl = dist.kl.kl_divergence(q, p)
    return torch.mean(torch.sum(kl, dim=1))


def reconstruction_loss(x_true, x_out):
    return F.mse_loss(x_true, x_out)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():   # GPU operations have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
