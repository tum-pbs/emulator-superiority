import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray


def individual_modes_gen(
    ic_config: str,
    num_spatial_dims: int,
):
    active_modes = ic_config.split(";")[1].split(",")
    max_one = ic_config.split(";")[2].lower() == "true"

    def generate(num_points: int, key: PRNGKeyArray):
        state_hat = jnp.zeros((1, num_points // 2 + 1), dtype=jnp.complex64)
        for mode in active_modes:
            mode = int(mode)
            subkey = jax.random.fold_in(key, mode)
            v = jax.random.normal(subkey, (), dtype=jnp.complex64)
            state_hat = state_hat.at[0, mode].set(v)

        state = jnp.fft.irfft(state_hat, n=num_points)
        if max_one:
            state = state / jnp.max(jnp.abs(state))
        return state

    return generate
