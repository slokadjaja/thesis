""" Analyze relation between patch distance (dtw or L2 norm) and encoding distance (hamming distance) """

import pandas as pd
from dataset import UCRDataset
from utils import *
from model import VAE
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
from scipy.spatial import distance
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


params_path = "../baseline_models/fc/params.json"
model_path = "../baseline_models/fc/model.pt"


def calc_distances():
    params = Params(params_path)

    dataset, patch_len, alphabet_size, n_latent, temperature, arch, normalize, norm_method = \
        params.dataset, params.patch_len, params.alphabet_size, params.n_latent, params.temperature, params.arch, \
        params.normalize, params.norm_method

    train = UCRDataset(dataset, "train", patch_len=patch_len, normalize=normalize, norm_method=norm_method)
    patches = train.x

    vae = VAE(patch_len, alphabet_size, n_latent, temperature, arch)
    vae.load_state_dict(torch.load(model_path))
    vae.eval()

    hamming_arr = []
    dtw_arr = []
    l2_arr = []

    comb = list(combinations(range(len(patches)), 2))
    total_iters = len(comb)

    with tqdm(total=total_iters, desc="Combinations: ") as pbar:
        for patch1, patch2 in combinations(patches, 2):
            patch1_np = patch1.squeeze().cpu().detach().numpy()
            patch2_np = patch2.squeeze().cpu().detach().numpy()
            dtw_arr.append(dtw(patch1_np, patch2_np))
            l2_arr.append(np.linalg.norm(patch2_np - patch1_np))

            enc1 = vae.encode(patch1).squeeze().cpu().detach().numpy()
            enc2 = vae.encode(patch2).squeeze().cpu().detach().numpy()
            hamming_arr.append(distance.hamming(enc1, enc2))

            pbar.update(1)

    df = pd.DataFrame()
    df['combination'] = comb
    df['hamming'] = hamming_arr
    df['dtw'] = dtw_arr
    df['l2'] = l2_arr

    scaler = MinMaxScaler()
    dtw_arr = np.array(dtw_arr).reshape(-1, 1)
    dtw_arr = list(scaler.fit_transform(dtw_arr).squeeze())
    l2_arr = np.array(l2_arr).reshape(-1, 1)
    l2_arr = list(scaler.fit_transform(l2_arr).squeeze())

    df['dtw_norm'] = dtw_arr
    df['l2_norm'] = l2_arr

    df.to_csv('distances.csv', index=False)


calc_distances()

data = pd.read_csv('distances.csv')

plt.scatter(data['hamming'], data['dtw'], s=0.2)
plt.title("hamming distance vs dtw")
plt.xlabel("hamming distance")
plt.ylabel("dtw")
plt.show()

plt.scatter(data['hamming'], data['l2'], s=0.2)
plt.title("hamming distance vs euclidean norm")
plt.xlabel("hamming distance")
plt.ylabel("euclidean norm")
plt.show()
