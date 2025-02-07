from base_encoder import BaseEncoder
import torch
import torch.nn.functional as F
from pathlib import Path
from utils import get_ts_length, Params
from model import VAE
import numpy as np

# copied from utils.py
def get_model_and_hyperparams(model_name: str, component=None) -> tuple[VAE, Params]:
    """Get a model in eval mode and hyperparameters used to train the model."""

    current_dir = Path(__file__).resolve().parent
    if component is not None:
        model_path = current_dir / "models" / model_name / component / "model.pt"
        params_path = current_dir / "models" / model_name / component / "params.json"
    else:
        model_path = current_dir / "models" / model_name / "model.pt"
        params_path = current_dir / "models" / model_name / "params.json"

    params = Params(params_path)
    ts_length = get_ts_length(params.dataset)
    input_dim = ts_length if params.patch_len is None else params.patch_len

    vae = VAE(input_dim=input_dim, alphabet_size=params.alphabet_size, n_latent=params.n_latent,
            temperature=params.temperature, model=params.arch)
    vae.load_state_dict(
        torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    )
    vae.eval()

    return vae, params


class VAEEncoder(BaseEncoder):
    def __init__(self, model_name, component=None):
        super().__init__()
        self.model, self.params = get_model_and_hyperparams(model_name=model_name, component=component)

    @property
    def patch_len(self):
        return get_ts_length(self.params.dataset) if self.params.patch_len is None else self.params.patch_len
    
    @property
    def alphabet_size(self):
        return self.params.alphabet_size

    # copied from utils.py, adjusted to interface
    def encode(self, data: np.ndarray):
        """Encodes entire time series datasets."""

        # input data shape: (batch, 1, len) or (batch, len)
        if data.shape[1] == 1:
            data = data.squeeze(axis=1)

        # Pad data according to patch_length
        ts_len = data.shape[-1]
        mod = ts_len % self.patch_length
        if mod != 0:
            data = np.pad(data, ((0, 0), (0, self.patch_length - mod)), 'constant')

        data_tensor = torch.Tensor(data)
        encoded_patches = []  # array to store encodings

        for i in range(0, data_tensor.shape[-1] - self.patch_length + 1, self.patch_length):
            window = data_tensor[:, i:i + self.patch_length]
            window = window.unsqueeze(1)  # Shape after: (batch, 1, patch_length)
            encoded_output = self.model.encode(window)

            # Remove the batch dimension if needed
            # encoded_output = encoded_output.squeeze(1)

            # Store the encoded output
            encoded_patches.append(encoded_output)

        # Stack all encoded patches into a tensor
        encoded_patches = torch.cat(encoded_patches, dim=1)
        encoded_patches_np = encoded_patches.numpy()

        assert len(encoded_patches_np.shape) == 2
        assert encoded_patches_np.shape[0] == data.shape[0]

        return encoded_patches_np