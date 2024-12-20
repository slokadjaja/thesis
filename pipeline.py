from train import Trainer
from hyp_tuning import tune_hyperparameters
from downstream_tasks.classification import classification
from utils import Params
from tqdm import tqdm
import itertools

if __name__ == "__main__":
    # todo set pipeline params externally?
    train_datasets = ["Wine", "ArrowHead", "p2s"]

    # Can also be found using optuna to save time
    patch_lens = [16, 32, 128]
    alphabet_sizes = [32, 48]

    use_azure = True
    cls_datasets = ["Wine", "Rock", "Plane", "ArrowHead", "p2s", "FordA", "FordB"]
    sax_params = [{"n_segments": 16, "alphabet_size": 32}, {"n_segments": 16, "alphabet_size": 48}]
    cls_trials = 10

    params = Params("params.json")
    run_names = []

    # todo also test arch, triplet loss
    # todo parallel
    for dataset, patch_len, alphabet_size in tqdm(itertools.product(train_datasets, patch_lens, alphabet_sizes),
                                                  total=len(train_datasets) * len(patch_lens) * len(alphabet_sizes),
                                                  desc="Pipeline progress"):
        # Just to save time
        if dataset == "p2s":
            if patch_len == 32 or alphabet_size == 48:
                continue
            params.epoch = 20
            hyp_tuning_trials = 3
        else:
            params.epoch = 200
            hyp_tuning_trials = 5

        run_name = f"{dataset}_p{patch_len}_a{alphabet_size}"
        print(f"\nStarting run: {run_name}")

        run_names.append(run_name)

        params.dataset = dataset
        params.patch_len = patch_len
        params.alphabet_size = alphabet_size

        print("\nStart tuning hyperparameters")
        params_tuned = tune_hyperparameters(params, run_name=run_name, azure=use_azure, n_trials=hyp_tuning_trials)

        print("\nHyperparameters tuned, start training using best hyperparameters")
        trainer = Trainer(params_tuned, run_name=run_name, azure=use_azure)
        trainer.train()

    print("\nStart classification")
    classification(datasets=cls_datasets, vae_models=run_names, sax_params=sax_params, iters_per_setting=cls_trials,
                   azure=use_azure)
