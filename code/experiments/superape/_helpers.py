from math import ceil
from typing import Callable

import equinox as eqx
import exponax as ex
import jax
import jax.numpy as jnp
import optax
from jax_tqdm import scan_tqdm
from jaxtyping import Array, Float, PRNGKeyArray, PyTree


def create_windowed_training_batches(
    trj_set: Float[Array, "num_trjs num_timesteps+1 ..."],
    *,
    window_size: int = 2,
    batch_size: int = 32,
    num_batches: int = 5000,
    print_info: bool = True,
    key: PRNGKeyArray,
) -> Float[Array, "num_batches batch_size window_size ..."]:
    """
    Turns a batch of trajectories into a series of windows that are randomly
    sampled across trajectories and across time.

    This is useful, for having training aligned in memory. It requires higher
    memory consumption (due to overlapping windows and repetition of data), but
    it is faster to train.

    **Arguments**:

    - `trj_set`: A batch of trajectories. The leading axis represents the batch
      dimensions, the following axis represents the number of snapshots. The
      axes thereafter depend on the kind of data. For example, these could be
      channel axes and spatial axes.
    - `window_size`: The size of the windows to be sampled. For one-step
      supervised learning, this should be 2 (to have an input and a target).
    - `batch_size`: The size of the batches to be returned, i.e., how many
      windows are in each batch.
    - `num_batches`: The number of batches to be returned. If the training is
      supposed to scan over this axis, this will equal the number of update
      steps performed. Note, the batches repeat after one epoch.
    - `print_info`: Whether to print information about the dataset, returning:
        - The number of windows in the dataset.
        - The number of effective epochs, i.e., how many times the dataset is
          repeated.
        - The number of batches per epoch.
    - `key`: A PRNG key for reproducible random shuffling.

    !!! info
        There are no remainder batches, i.e., batches with a different batch
        size. The dataset is repeated/cycled such that a potential remainder
        batch is filled with samples from the next epoch.

    """
    # S: number of trajectories, T: number of timesteps, W: window size, N:
    # number of windows, U: number of batches, B: batch size, E: number of
    # epochs (ceiled),

    # (S, T+1, ...) -> (S, T-W+2, W, ...)
    substacked_trj_set = jax.vmap(ex.stack_sub_trajectories, in_axes=(0, None))(
        trj_set, window_size
    )
    # (W, T-W+2, W, ...) -> (N, W, ...)  (with N = S * (T-W+2))
    window_set = jnp.concatenate(substacked_trj_set)
    num_windows = window_set.shape[0]

    # E = ceil(U * B / N)
    num_epochs = ceil(num_batches * batch_size / num_windows)

    # An array or repeated permutations of the window indices, (E, N)
    permutations = jax.vmap(jax.random.permutation, in_axes=(0, None))(
        jax.random.split(key, num_epochs), jnp.arange(num_windows)
    )
    # (E, N) -> (E * N,)
    concated_permutations = jnp.concatenate(permutations)

    def scan_fn(_, i):
        permutes = jax.lax.dynamic_slice_in_dim(
            concated_permutations, i * batch_size, batch_size
        )
        return None, window_set[permutes]

    # (N, W, ...) -> (U, B, W, ...)  (with U*B <= N)
    _, sliced_window_set = jax.lax.scan(scan_fn, None, jnp.arange(num_batches))

    num_effective_epochs = num_batches * batch_size / num_windows
    batches_per_epoch = num_windows / batch_size

    if print_info:
        print(
            f"Number of windows: {num_windows}, "
            f"Number of effective epochs: {num_effective_epochs:.2f}, "
            f"Batches per epoch: {batches_per_epoch:.2f}"
        )

    return jnp.stack(sliced_window_set)


def train_scanned(
    model: eqx.Module,
    data: PyTree[Float[Array, "num_minibatches batch_size ..."]],
    optimizer: optax.GradientTransformation,
    loss_fn: Callable[[eqx.Module, PyTree[Float[Array, "batch_size ..."]]], float],
    metric_fn: Callable[
        [
            eqx.Module,
        ],
        PyTree[Float[Array, "..."]],
    ],
    opt_state=None,
    *,
    print_rate: int = 100,
) -> tuple[
    eqx.Module,
    optax.OptState,
    Float[Array, " num_minibatches "],
    PyTree[Float[Array, " num_minibatches ..."]],
]:
    """
    Trains an equinox model while scanning over a dataset. This is more
    efficient than a Python for loop if the data can preprocessed accordingly
    and if there are expensive `metric_fn` calls.

    !!! tip
        You can preprocess the data with `create_windowed_training_batches`.

    **Arguments**:

    - `model`: The model to be trained, with its initial parameters.
    - `data`: The training data, a PyTree of arrays. The leading axis of the
        arrays represents the number of minibatches, corresponding to the number
        of steps performed in the scan (and the number of update steps). The
        following axis represents the batch size and the axes thereafter depend
        on the kind of data. For example, these could be channel axes and spatial
        axes.
    - `optimizer`: The optimizer to be used for training.
    - `loss_fn`: The loss function to be minimized. It takes the current model
        and a batch of data and returns a scalar loss.
    - `metric_fn`: A function that takes the current model and returns a PyTree
        of metrics. This is useful for monitoring the training progress.
    - `opt_state`: The initial optimizer state. If `None`, the optimizer is
        initialized with the parameters of the model.
    - `print_rate`: The rate at which the progress bar is updated.

    **Returns**:

    - The model with the final parameters.
    - The final optimizer state (e.g., can be used for further training).
    - The losses at each minibatch.
    - A PyTree of metrics at each minibatch. The arrays within the PyTree have
        the shape `(num_minibatches, ...)`.
    """
    params, constants = eqx.partition(model, eqx.is_array)
    if opt_state is None:
        opt_state = optimizer.init(params)

    def wrapped_loss_fn(params, batch):
        current_model = eqx.combine(params, constants)
        return loss_fn(current_model, batch)

    def wrapped_metric_fn(params):
        current_model = eqx.combine(params, constants)
        return metric_fn(current_model)

    def update_fn(params, state, batch):
        loss, grads = jax.value_and_grad(wrapped_loss_fn)(params, batch)
        updates, new_state = optimizer.update(grads, state)
        new_params = optax.apply_updates(params, updates)
        metric = wrapped_metric_fn(new_params)
        return new_params, new_state, loss, metric

    num_training_steps = data.shape[0]

    @scan_tqdm(num_training_steps, print_rate=print_rate)
    def scan_fn(carry, i):
        batch = data[i]
        params, state = carry
        new_params, new_state, loss, metric = update_fn(params, state, batch)
        return (new_params, new_state), (loss, metric)

    (final_params, final_state), (losses, metrics) = jax.lax.scan(
        scan_fn, (params, opt_state), jnp.arange(num_training_steps)
    )

    final_model = eqx.combine(final_params, constants)

    return final_model, final_state, losses, metrics
