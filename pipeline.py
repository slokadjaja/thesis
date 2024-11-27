from train import Trainer
from hyp_tuning import tune_hyperparameters
from downstream_tasks.classification import classification
from utils import Params
import itertools

if __name__ == "__main__":
    datasets = ["Wine", "ArrowHead", "p2s"]
    patch_lens = [16, 32, 128]
    alphabet_sizes = [32, 48]

    params = Params("params.json")
    run_names = []

    # todo also test arch
    for dataset, patch_len, alphabet_size in itertools.product(datasets, patch_lens, alphabet_sizes):
        run_name = f"{dataset}_p{patch_len}_a{alphabet_size}"
        params.run_name = run_name
        run_names.append(run_name)

        params.dataset = dataset
        params.patch_len = patch_len
        params.alphabet_size = alphabet_size

        params_tuned = tune_hyperparameters(params, run_name=run_name)  # Hyperparameter tuning
        trainer = Trainer(params_tuned)     # Train model using optimal hyperparameters
        trainer.train()

    # Test trained models on classification
    dataset_list = ["Wine", "Rock", "Plane", "ArrowHead", "p2s"]
    sax_list = [{"n_segments": 128, "alphabet_size": 32}, {"n_segments": 128, "alphabet_size": 48}]
    n_iters = 10

    classification(dataset_list, run_names, sax_list, n_iters)
