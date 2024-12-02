from train import Trainer
from hyp_tuning import tune_hyperparameters
from downstream_tasks.classification import classification
from utils import Params
from tqdm import tqdm
import itertools

if __name__ == "__main__":
    datasets = ["Wine", "ArrowHead", "p2s"]
    patch_lens = [16, 32, 128]
    alphabet_sizes = [32, 48]

    params = Params("params.json")
    run_names = []

    # todo also test arch
    for dataset, patch_len, alphabet_size in tqdm(itertools.product(datasets, patch_lens, alphabet_sizes),
                                                  total=len(datasets) * len(patch_lens) * len(alphabet_sizes),
                                                  desc="Pipeline progress"):
        run_name = f"{dataset}_p{patch_len}_a{alphabet_size}"
        print(f"Starting run: {run_name}")

        run_names.append(run_name)

        params.dataset = dataset
        params.patch_len = patch_len
        params.alphabet_size = alphabet_size

        print("Start tuning hyperparameters")
        params_tuned = tune_hyperparameters(params, run_name=run_name)  # Hyperparameter tuning

        print("Hyperparameters tuned, start training using best hyperparameters")
        trainer = Trainer(params_tuned, run_name=run_name)  # Train model using optimal hyperparameters
        trainer.train()

    print("Start classification")
    # Test trained models on classification
    dataset_list = ["Wine", "Rock", "Plane", "ArrowHead", "p2s"]
    sax_list = [{"n_segments": 128, "alphabet_size": 32}, {"n_segments": 128, "alphabet_size": 48}]
    n_iters = 10

    classification(dataset_list, run_names, sax_list, n_iters)
