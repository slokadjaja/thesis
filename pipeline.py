from train import Trainer
from hyp_tuning import tune_hyperparameters
from downstream_tasks.classification import classification
from utils import Params
from tqdm import tqdm
import itertools

if __name__ == "__main__":
    # todo set pipeline params externally?
    train_datasets = ["Wine", "ArrowHead", "p2s"]
    patch_lens = [16, 32, 128]
    alphabet_sizes = [32, 48]
    use_azure = True
    hyp_tuning_trials = 10
    cls_datasets = ["Wine", "Rock", "Plane", "ArrowHead", "p2s", "FordA", "FordB"]
    sax_params = [{"n_segments": 128, "alphabet_size": 32}, {"n_segments": 128, "alphabet_size": 48}]
    cls_trials = 10

    params = Params("params.json")
    run_names = []

    # todo also test arch, triplet loss
    for dataset, patch_len, alphabet_size in tqdm(itertools.product(train_datasets, patch_lens, alphabet_sizes),
                                                  total=len(train_datasets) * len(patch_lens) * len(alphabet_sizes),
                                                  desc="Pipeline progress"):
        run_name = f"{dataset}_p{patch_len}_a{alphabet_size}"
        print(f"\nStarting run: {run_name}")

        run_names.append(run_name)

        params.dataset = dataset
        params.patch_len = patch_len
        params.alphabet_size = alphabet_size

        print("Start tuning hyperparameters")
        params_tuned = tune_hyperparameters(params, run_name=run_name, azure=use_azure, n_trials=hyp_tuning_trials)

        print("Hyperparameters tuned, start training using best hyperparameters")
        trainer = Trainer(params_tuned, run_name=run_name, azure=use_azure)
        trainer.train()

    print("Start classification")
    classification(datasets=cls_datasets, vae_models=run_names, sax_params=sax_params, iters_per_setting=cls_trials)
