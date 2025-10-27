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
    *,
    network,
    gamma_2,
    delta_1,
    num_points,
    ic_distribution,
    p_iters,
    seed_init,
    seed_train,
    seed_test,
    seed_shuffle,
):
    config = {
        "network": network,
        "linear_difficulties": (0.0, 0.0, gamma_2, 0.0, 0.0),
        "convection_difficulty": delta_1,
        "num_points": num_points,
        "simulator_train": f"fd;1.0;{p_iters}",  # Implicit FOU with 1 Picard step
        "simulator_test": "fd;1.0;100",  # Implicit FOU with up to 100 Picard steps
        "ic_distribution_train": ic_distribution,
        "ic_distribution_test": ic_distribution,
        "temporal_horizon_train": 1,
        "temporal_horizon_test": 10,
        "num_warmup_steps": 0,
        "num_unrolled_steps": 1,
        "metric_fn_train": "mean_MSE",
        "metric_fn_test": "mean_nRMSE",
        "num_samples_train": 500,
        "num_samples_test": 20,
        "optimizer": "adam;100_000;warmup_cosine;0.0;3e-4;5_000",
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
            "gamma_2": gamma_2,
            "delta_1": delta_1,
            "num_points": num_points,
            "ic_distribution": ic_distribution,
            "p_iters": p_iters,
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
            "gamma_2": gamma_2,
            "delta_1": delta_1,
            "num_points": num_points,
            "ic_distribution": ic_distribution,
            "p_iters": p_iters,
            "seed_init": seed_init,
            "seed_train": seed_train,
            "seed_test": seed_test,
            "seed_shuffle": seed_shuffle,
            "loss": losses,
            "update_step": jnp.arange(1, len(losses) + 1),
        }
    )
    return trained_net, data_metrics, data_loss, trj_dict

# def get_trj(
#     setting: dict,
# ):
#     os.makedirs("burgers_trj", exist_ok=True)
#     setting_str = json.dumps(setting)
#     if os.path.exists(f"burgers_trj/{setting_str}"):
#         trj_dict = jnp.load(f"burgers_trj/{setting_str}/trj.npz")
#     else:
#         os.makedirs(f"burgers_trj/{setting_str}", exist_ok=True)
#         _, _, _, trj_dict = get_data(**setting)

#         jnp.savez(
#             f"burgers_trj/{setting_str}/trj.npz",
#             **trj_dict,
#         )

#     return trj_dict

def get_all_data(
    settings: list[dict],
) -> list[pd.DataFrame]:
    os.makedirs("burgers_data_varying_picard_iters", exist_ok=True)
    data_metrics = []
    data_loss = []
    for setting in tqdm(settings):
        setting_str = json.dumps(setting)
        if os.path.exists(f"burgers_data_varying_picard_iters/{setting_str}"):
            data_metrics.append(
                pd.read_csv(f"burgers_data_varying_picard_iters/{setting_str}/metrics.csv.gz", compression="gzip")
            )
            data_loss.append(
                pd.read_csv(f"burgers_data_varying_picard_iters/{setting_str}/loss.csv.gz", compression="gzip")
            )
        else:
            os.makedirs(f"burgers_data_varying_picard_iters/{setting_str}", exist_ok=True)
            _, dm, dl, _ = get_data(**setting)
            dm.to_csv(
                f"burgers_data_varying_picard_iters/{setting_str}/metrics.csv.gz", index=False, compression="gzip"
            )
            dl.to_csv(f"burgers_data_varying_picard_iters/{setting_str}/loss.csv.gz", index=False, compression="gzip")
            data_metrics.append(dm)
            data_loss.append(dl)
    return data_metrics, data_loss
