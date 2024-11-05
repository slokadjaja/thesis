import torch
from torch.utils.data import DataLoader
from dataset import UCRDataset
from utils import get_ts_length, cat_kl_div, reconstruction_loss, set_seed, Params, contrastive_loss
from model import VAE
from tqdm import tqdm
import mlflow
import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)


def champion_callback(study, frozen_trial):
    """Logging callback that will report when a new trial iteration improves upon existing best trial values."""
    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


def get_or_create_experiment(experiment_name):
    """Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist."""
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


def objective(trial):
    params = Params("params.json")

    # Define constants
    if params.seed:
        set_seed(params.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts_length = get_ts_length(params.dataset)
    input_dim = ts_length if params.patch_len is None else params.patch_len

    with mlflow.start_run(nested=True):
        # Specify hyperparams to test
        params.lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        params.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        params.temperature = trial.suggest_float('temperature', 0.2, 1.5, step=0.1)

        mlflow.log_params(params.dict)

        # Load dataset
        train = UCRDataset(params.dataset, "train", patch_len=params.patch_len, normalize=params.normalize,
                           norm_method=params.norm_method)
        # shape of batch: [batch_size, 1, length]
        train_dataloader = DataLoader(train, batch_size=params.batch_size, shuffle=True)

        # Define and train model
        vae = VAE(input_dim, params.alphabet_size, params.n_latent, params.temperature, params.arch).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=params.lr)

        # Training loop
        vae.train()

        total_loss = 0
        for epoch in tqdm(range(params.epoch), desc="Epoch"):
            epoch_loss = 0
            for x, y in train_dataloader:
                x = x.to(device)

                logits, output = vae(x)
                rec_loss = reconstruction_loss(torch.squeeze(x, dim=1), torch.squeeze(output, dim=1))
                kl_div = cat_kl_div(logits, n_latent=params.n_latent, alphabet_size=params.alphabet_size)
                closs = contrastive_loss(x, params.top_quantile, params.bottom_quantile, params.margin)

                loss = rec_loss + params.beta * kl_div + params.alpha * closs
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Log the average loss per epoch
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            total_loss += avg_epoch_loss

            mlflow.log_metrics(
                {
                    "total loss": loss.item(), "reconstruction loss": rec_loss.item(), "kl divergence": kl_div.item(),
                    "contrastive loss": closs.item(), "avg_epoch_loss": avg_epoch_loss
                }, step=epoch
            )

            trial.report(avg_epoch_loss, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # return last epoch loss
    return avg_epoch_loss


if __name__ == "__main__":
    # Set mlflow experiment
    experiment_id = get_or_create_experiment("hyperparams_tuning")
    mlflow.set_experiment(experiment_id=experiment_id)
    n_trials = 5

    with mlflow.start_run(experiment_id=experiment_id, run_name="test", nested=True):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, callbacks=[champion_callback])

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_loss", study.best_value)
        print(f"Best trial parameters: {study.best_params}")
        print(f"Best trial final loss: {study.best_value}")

        # todo
        # what does best_params look like? how to save it alongside untested params
        # log model params (model.pt)? architecture? reconstruction plots?
