import json
import os

import apebench
import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm

import superape
from individual_modes_ic import individual_modes_gen
from transformer import transformer_constructor

apebench.components.architecture_dict["trans"] = transformer_constructor
apebench.components.ic_dict["ind"] = individual_modes_gen


def get_data(
    network,
    gamma_1,
    train_simulator,
    train_ic_distribution,
    test_ic_distribution,
    seed_init,
    seed_train,
    seed_test,
    seed_shuffle,
):
    config = {
        "network": network,
        "linear_difficulties": (0.0, gamma_1, 0.0, 0.0, 0.0),
        "convection_difficulty": 0.0,
        "num_points": 100,
        "simulator_train": train_simulator,
        "simulator_test": "etdrk;2",
        "ic_distribution_train": train_ic_distribution,
        "ic_distribution_test": test_ic_distribution,
        "temporal_horizon_train": 1,
        "temporal_horizon_test": 5000,
        "num_warmup_steps": 0,
        "metric_fn_train": "mean_MSE",
        "metric_fn_test": "mean_RMSE",
        "num_samples_train": 300,
        "num_samples_test": 10,
        "num_unrolled_steps": 1,
        "optimizer": "adam;10_000;warmup_cosine;0.0;1e-3;1_000",
        "batch_size": 32,
        "seed_init": seed_init,
        "seed_train": seed_train,
        "seed_test": seed_test,
        "seed_shuffle": seed_shuffle,
    }
    trained_net, losses, metrics, trj_dict = superape.run_no_recording(**config)
    superiority_trj = metrics["superiority_trj"]
    data_metrics = pd.DataFrame(
        {
            "network": network,
            "gamma_1": gamma_1,
            "train_simulator": train_simulator,
            "train_ic_distribution": train_ic_distribution,
            "test_ic_distribution": test_ic_distribution,
            "seed_init": seed_init,
            "seed_train": seed_train,
            "seed_test": seed_test,
            "seed_shuffle": seed_shuffle,
            "superiority": superiority_trj,
            "error": metrics["error_trj"],
            "time_step": jnp.arange(1, len(superiority_trj) + 1),
        }
    )
    data_loss = pd.DataFrame(
        {
            "network": network,
            "gamma_1": gamma_1,
            "train_simulator": train_simulator,
            "train_ic_distribution": train_ic_distribution,
            "test_ic_distribution": test_ic_distribution,
            "seed_init": seed_init,
            "seed_train": seed_train,
            "seed_test": seed_test,
            "seed_shuffle": seed_shuffle,
            "loss": losses,
            "update_step": jnp.arange(1, len(losses) + 1),
        }
    )
    return data_metrics, data_loss


def get_all_data(
    settings: list[dict],
) -> list[pd.DataFrame]:
    os.makedirs("advection_data", exist_ok=True)
    data_metrics = []
    data_loss = []
    for setting in tqdm(settings):
        setting_str = json.dumps(setting)
        if os.path.exists(f"advection_data/{setting_str}"):
            data_metrics.append(
                pd.read_csv(f"advection_data/{setting_str}/metrics.csv.gz", compression="gzip")
            )
            data_loss.append(
                pd.read_csv(f"advection_data/{setting_str}/loss.csv.gz", compression="gzip")
            )
        else:
            os.makedirs(f"advection_data/{setting_str}", exist_ok=True)
            dm, dl = get_data(**setting)
            # Compress the data
            dm.to_csv(
                f"advection_data/{setting_str}/metrics.csv.gz", index=False, compression="gzip"
            )
            dl.to_csv(
                f"advection_data/{setting_str}/loss.csv.gz", index=False, compression="gzip"
            )
            data_metrics.append(dm)
            data_loss.append(dl)
    return data_metrics, data_loss
