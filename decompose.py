from train import Trainer
from utils import Params, get_or_create_experiment, get_model_and_hyperparams, vae_encoding
import mlflow
from tqdm import tqdm
import numpy as np
from utils import get_dataset


def train_multiple(params, exp_name, run_name, use_azure, components):
    """Train multiple vae models, each corresponding to a different time series component."""
    trainers = []
    
    experiment_id = get_or_create_experiment(exp_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        for component in tqdm(components, desc="Processing components"):
            print(f"Component: {component}")
            trainers.append(
                Trainer(
                    params=params,
                    experiment_name=exp_name,
                    run_name=run_name,
                    azure=use_azure,
                    component=component,
                )
            )
            trainers[-1].train()


def encode_multiple(data: np.ndarray, model_name: str, components: list[str]):
    """Encode time series data using multiple models, each corresponding to a different time series component."""
    all_encodings = []
    
    for component in components:
        vae, params = get_model_and_hyperparams(model_name, component=component)
        encoding = vae_encoding(vae, data)
        all_encodings.append(encoding)

    final_encoding = np.concatenate(all_encodings, axis=1)
    return final_encoding


if __name__ == "__main__":
    params = Params("params.json")
    exp_name = "decompose"
    run_name = f"{params.dataset}_p{params.patch_len}_a{params.alphabet_size}"
    use_azure = False
    components = ["trend", "seasonality", "residual"]
    
    X_train, y_train, X_test, y_test = get_dataset("Wine")
    encoding = encode_multiple(X_train, run_name, components)