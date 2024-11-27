import json
from train import Trainer
from utils import Params
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

def tune_hyperparameters(params, target_path=None, n_trials=10, experiment_name="hyperparams_tuning", run_name="test"):
    def objective(trial):
        # Suggest hyperparameters
        params.lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        params.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        params.temperature = trial.suggest_float('temperature', 0.2, 1.5, step=0.1)
        
        trainer = Trainer(params)
        with mlflow.start_run(nested=True):
            mlflow.log_params(params.dict)

            total_loss = 0
            for epoch in tqdm(range(params.epoch), desc="Epoch"):
                loss, rec_loss, kl_div, closs = trainer.train_one_epoch()
                mlflow.log_metrics({"total loss": loss, "reconstruction loss": rec_loss, "kl divergence": kl_div,
                                    "contrastive loss": closs}, step=epoch)

                total_loss += loss
                avg_loss = total_loss/(epoch+1)     # Report average loss per epoch so far

                # todo use validation loss?

                # Handle pruning based on the intermediate value.
                trial.report(avg_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        return avg_loss

    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, callbacks=[champion_callback])

        # Save best hyperparameters to JSON
        best_params = study.best_params
        best_loss = study.best_value

        params.dict.update(best_params)
        if target_path:
            with open(target_path, "w") as f:
                json.dump(params.dict, f)

        mlflow.log_params(best_params)
        mlflow.log_metric("best_loss", best_loss)
        
        print("Best hyperparameters:", best_params)
        print(f"Best final loss: {best_loss}")

    return params


if __name__ == "__main__":
    params = Params("params.json")
    params_tuned = tune_hyperparameters(params,"params_tuned.json")