import apebench
import equinox as eqx
import exponax as ex
import jax
import trainax as tx
from jaxtyping import Array, Float

from ._helpers import create_windowed_training_batches, train_scanned
from ._parser import parse_simulator


def run(
    *,
    network: str,
    linear_difficulties: tuple[float, float, float, float, float],
    convection_difficulty: float,
    num_points: int,
    simulator_train: str,
    simulator_test: str,
    ic_distribution_train: str,
    ic_distribution_test: str,
    temporal_horizon_train: int,
    temporal_horizon_test: int,
    num_warmup_steps: int,
    metric_fn_train: str,
    metric_fn_test: str,
    num_samples_train: int,
    num_samples_test: int,
    num_unrolled_steps: int,
    optimizer: str,
    batch_size: int,
    seed_init: int,
    seed_train: int,
    seed_test: int,
    seed_shuffle: int,
) -> tuple[
    eqx.Module,
    Float[Array, " num_training_steps "],
    dict[str, Float[Array, " num_training_steps temporal_horizon_test "]],
    dict[str, Float[Array, " num_samples_test temporal_horizon_test 1 num_points "]],
]:
    """
    Given a keyword-based configuration, produce a loss history array and an
    metric rollout array (over the training)

    Train trajectories will be of length `temporal_horizon_train + 1`, whereas
    test trajectories will be of length `temporal_horizon_test`

    !!! example
        **Emulator superiority** in advection emulation

        ```python
        net, losses, metric, trjs = new_evaluation_interface(
            network="Conv;34;10;relu",
            linear_difficulties=(0.0, -4.0, 0.0, 0.0, 0.0),
            convection_difficulty=0.0,
            num_points=100,
            simulator_train="fd;1.0;100",
            simulator_test="etdrk;2",
            ic_distribution_train="fourier;2;2;True",
            ic_distribution_test="fourier;2;2;True",
            temporal_horizon_train=50,
            temporal_horizon_test=30,
            num_warmup_steps=0,
            metric_fn_train="mean_MSE",
            metric_fn_test="mean_MSE",
            num_samples_train=20,
            num_samples_test=10,
            num_unrolled_steps=1,
            optimizer="adam;1_000;constant;3e-4",
            batch_size=20,
            seed_init=0,
            seed_train=1,
            seed_test=2,
            seed_shuffle=3,
        )
        ```

        And then

        ```python

        plt.plot(metric["superiority_trj"][-1])
        plt.xlabel("Time Step")
        plt.ylabel("Superiority")
        plt.hlines(1, 0, 30, colors="black", linestyles="dashed")

        ```

        reveals

        ![Advection emulator superiority simple
        example](https://github.com/user-attachments/assets/55e8ac22-4857-46fe-81d9-28d87e9c7c72)

    !!! tip
        Perform seed experiments. First, wrap your settings

        ```python

        def wrapped_eval(seed_init, seed_train, seed_test, seed_shuffle):
            return new_evaluation_interface(
                # ...
                seed_init=seed_init,
                seed_train=seed_train,
                seed_test=seed_test,
                seed_shuffle=seed_shuffle,
            )

        ```

        Then use

        ```python

        def seed_experiment(wrapped_fn):
            vmapped_fn = eqx.filter_vmap(
                eqx.filter_vmap(
                    eqx.filter_vmap(
                        eqx.filter_vmap(
                            wrapped_fn,
                            in_axes=(None, None, None, 0),
                        ),
                        in_axes=(None, None, 0, None),
                    ),
                    in_axes=(None, 0, None, None),
                ),
                in_axes=(0, None, None, None),
            )
            return vmapped_fn

        ```

        And run (for example with a cross-combination on the different seeds)

        ```python

        net_s, losses_s, metric_s, trjs_s = seed_experiment(wrapped_eval)(
            jnp.arange(0, 5),
            jnp.arange(5, 9),
            jnp.arange(9, 12),
            jnp.arange(12, 14),
        )

        ```

        Take care to not overload your GPU too much. Beyond a certain number of
        seeds, it is more efficient to run them sequentially.
    """

    # Parse simulators
    train_simulator = parse_simulator(
        simulator_train, num_points, linear_difficulties, convection_difficulty
    )
    test_simulator = parse_simulator(
        simulator_test, num_points, linear_difficulties, convection_difficulty
    )

    # Parse IC distributions
    train_ic_distribution = apebench.components.ic_dict[
        ic_distribution_train.split(";")[0].lower()
    ](ic_distribution_train, 1)
    test_ic_distribution = apebench.components.ic_dict[
        ic_distribution_test.split(";")[0].lower()
    ](ic_distribution_test, 1)

    # Parse metric functions
    metric_fn_train = apebench.components.metric_dict[metric_fn_train.split(";")[0]](
        metric_fn_train
    )
    metric_fn_test = apebench.components.metric_dict[metric_fn_test.split(";")[0]](
        metric_fn_test
    )

    # Parse optimizer
    optimizer_name = optimizer.split(";")[0].lower()
    num_training_steps = int(optimizer.split(";")[1])
    lr_scheduler_name = optimizer.split(";")[2].lower()
    lr_scheduler_config = ";".join(optimizer.split(";")[2:])
    lr_scheduler = apebench.components.lr_scheduler_dict[lr_scheduler_name](
        lr_scheduler_config, num_training_steps
    )
    optimizer = apebench.components.optimizer_dict[optimizer_name](optimizer)(
        lr_scheduler
    )

    # Parse network
    network_name = network.split(";")[0].lower()
    activation_fn = apebench.components.activation_fn_dict[
        network.split(";")[-1].lower()
    ]("")
    network = apebench.components.architecture_dict[network_name](
        network, 1, num_points, 1, activation_fn, jax.random.key(seed_init)
    )

    # Prepare data
    key_train = jax.random.key(seed_train)
    key_test = jax.random.key(seed_test)
    key_shuffle = jax.random.key(seed_shuffle)

    train_ic_set = ex.build_ic_set(
        train_ic_distribution,
        num_points=num_points,
        num_samples=num_samples_train,
        key=key_train,
    )
    train_ic_set = jax.vmap(ex.repeat(test_simulator, num_warmup_steps))(train_ic_set)
    test_ic_set = ex.build_ic_set(
        test_ic_distribution,
        num_points=num_points,
        num_samples=num_samples_test,
        key=key_test,
    )
    test_ic_set = jax.vmap(ex.repeat(test_simulator, num_warmup_steps))(test_ic_set)

    train_trj_set = jax.vmap(
        ex.rollout(train_simulator, temporal_horizon_train, include_init=True)
    )(train_ic_set)

    minibatches = create_windowed_training_batches(
        train_trj_set,
        window_size=num_unrolled_steps + 1,
        batch_size=batch_size,
        num_batches=num_training_steps,
        key=key_shuffle,
    )

    test_trj_set = jax.vmap(
        ex.rollout(test_simulator, temporal_horizon_test, include_init=False)
    )(test_ic_set)
    test_trj_set_using_train_simulator = jax.vmap(
        ex.rollout(train_simulator, temporal_horizon_test, include_init=False)
    )(test_ic_set)

    def error_trj_against_test_simulator(stepper):
        def scan_fn(state, ref_next_state):
            next_state = jax.vmap(stepper)(state)
            error = metric_fn_test(next_state, ref_next_state)
            return next_state, error

        _, error_trj = jax.lax.scan(
            scan_fn, test_ic_set, test_trj_set.transpose(1, 0, 2, 3)
        )

        return error_trj

    error_trj_train_simulator = error_trj_against_test_simulator(train_simulator)

    def test_callback_fn(model):
        error_trj = error_trj_against_test_simulator(model)
        superiority_trj = error_trj / error_trj_train_simulator
        return {
            "error_trj": error_trj,
            "superiority_trj": superiority_trj,
        }

    trained_network, _, losses, metrics = train_scanned(
        network,
        minibatches,
        optimizer,
        tx.configuration.Supervised(
            num_unrolled_steps, time_level_loss=metric_fn_train
        ),
        test_callback_fn,
    )

    test_trj_set_neural = jax.vmap(
        ex.rollout(trained_network, temporal_horizon_test, include_init=False)
    )(test_ic_set)

    trj_dict = {
        "train": train_trj_set,
        "test": test_trj_set,
        "test_using_train_simulator": test_trj_set_using_train_simulator,
        "test_neural": test_trj_set_neural,
    }

    return trained_network, losses, metrics, trj_dict


def run_no_recording(
    *,
    network: str,
    linear_difficulties: tuple[float, float, float, float, float],
    convection_difficulty: float,
    num_points: int,
    simulator_train: str,
    simulator_test: str,
    ic_distribution_train: str,
    ic_distribution_test: str,
    temporal_horizon_train: int,
    temporal_horizon_test: int,
    num_warmup_steps: int,
    metric_fn_train: str,
    metric_fn_test: str,
    num_samples_train: int,
    num_samples_test: int,
    num_unrolled_steps: int,
    optimizer: str,
    batch_size: int,
    seed_init: int,
    seed_train: int,
    seed_test: int,
    seed_shuffle: int,
) -> tuple[
    eqx.Module,
    Float[Array, " num_training_steps "],
    dict[str, Float[Array, " num_training_steps temporal_horizon_test "]],
    dict[str, Float[Array, " num_samples_test temporal_horizon_test 1 num_points "]],
]:
    # Parse simulators
    train_simulator = parse_simulator(
        simulator_train, num_points, linear_difficulties, convection_difficulty
    )
    test_simulator = parse_simulator(
        simulator_test, num_points, linear_difficulties, convection_difficulty
    )

    # Parse IC distributions
    train_ic_distribution = apebench.components.ic_dict[
        ic_distribution_train.split(";")[0].lower()
    ](ic_distribution_train, 1)
    test_ic_distribution = apebench.components.ic_dict[
        ic_distribution_test.split(";")[0].lower()
    ](ic_distribution_test, 1)

    # Parse metric functions
    metric_fn_train = apebench.components.metric_dict[metric_fn_train.split(";")[0]](
        metric_fn_train
    )
    metric_fn_test = apebench.components.metric_dict[metric_fn_test.split(";")[0]](
        metric_fn_test
    )

    # Parse optimizer
    optimizer_name = optimizer.split(";")[0].lower()
    num_training_steps = int(optimizer.split(";")[1])
    lr_scheduler_name = optimizer.split(";")[2].lower()
    lr_scheduler_config = ";".join(optimizer.split(";")[2:])
    lr_scheduler = apebench.components.lr_scheduler_dict[lr_scheduler_name](
        lr_scheduler_config, num_training_steps
    )
    optimizer = apebench.components.optimizer_dict[optimizer_name](optimizer)(
        lr_scheduler
    )

    # Parse network
    network_name = network.split(";")[0].lower()
    activation_fn = apebench.components.activation_fn_dict[
        network.split(";")[-1].lower()
    ]("")
    network = apebench.components.architecture_dict[network_name](
        network, 1, num_points, 1, activation_fn, jax.random.key(seed_init)
    )

    # Prepare data
    key_train = jax.random.key(seed_train)
    key_test = jax.random.key(seed_test)
    key_shuffle = jax.random.key(seed_shuffle)

    train_ic_set = ex.build_ic_set(
        train_ic_distribution,
        num_points=num_points,
        num_samples=num_samples_train,
        key=key_train,
    )
    train_ic_set = jax.vmap(ex.repeat(test_simulator, num_warmup_steps))(train_ic_set)
    test_ic_set = ex.build_ic_set(
        test_ic_distribution,
        num_points=num_points,
        num_samples=num_samples_test,
        key=key_test,
    )
    test_ic_set = jax.vmap(ex.repeat(test_simulator, num_warmup_steps))(test_ic_set)

    train_trj_set = jax.vmap(
        ex.rollout(train_simulator, temporal_horizon_train, include_init=True)
    )(train_ic_set)

    minibatches = create_windowed_training_batches(
        train_trj_set,
        window_size=num_unrolled_steps + 1,
        batch_size=batch_size,
        num_batches=num_training_steps,
        key=key_shuffle,
    )

    test_trj_set = jax.vmap(
        ex.rollout(test_simulator, temporal_horizon_test, include_init=False)
    )(test_ic_set)
    test_trj_set_using_train_simulator = jax.vmap(
        ex.rollout(train_simulator, temporal_horizon_test, include_init=False)
    )(test_ic_set)

    def error_trj_against_test_simulator(stepper):
        def scan_fn(state, ref_next_state):
            next_state = jax.vmap(stepper)(state)
            error = metric_fn_test(next_state, ref_next_state)
            return next_state, error

        _, error_trj = jax.lax.scan(
            scan_fn, test_ic_set, test_trj_set.transpose(1, 0, 2, 3)
        )

        return error_trj

    error_trj_train_simulator = error_trj_against_test_simulator(train_simulator)

    def test_callback_fn(model):
        error_trj = error_trj_against_test_simulator(model)
        superiority_trj = error_trj / error_trj_train_simulator
        return {
            "error_trj": error_trj,
            "superiority_trj": superiority_trj,
        }

    trained_network, _, losses, _ = train_scanned(
        network,
        minibatches,
        optimizer,
        tx.configuration.Supervised(
            num_unrolled_steps, time_level_loss=metric_fn_train
        ),
        lambda model: None,
    )

    test_trj_set_neural = jax.vmap(
        ex.rollout(trained_network, temporal_horizon_test, include_init=False)
    )(test_ic_set)

    trj_dict = {
        "train": train_trj_set,
        "test": test_trj_set,
        "test_using_train_simulator": test_trj_set_using_train_simulator,
        "test_neural": test_trj_set_neural,
    }

    metrics = test_callback_fn(trained_network)

    return trained_network, losses, metrics, trj_dict
