from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


def scaled_dot_product_attention(
    query: Float[Array, "seq_len key_size"],
    key: Float[Array, "seq_len key_size"],
    value: Float[Array, "seq_len value_size"],
) -> Float[Array, "seq_len value_size"]:
    """No masking"""
    key_size = query.shape[-1]
    attn_logits: Float[Array, "seq_len seq_len"] = (query @ key.T) / jnp.sqrt(key_size)
    attn_weights: Float[Array, "seq_len seq_len"] = jax.nn.softmax(attn_logits, axis=-1)
    results: Float[Array, "seq_len value_size"] = attn_weights @ value
    return results


class MultiheadAttention(eqx.Module):
    num_heads: int
    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        *,
        key: PRNGKeyArray,
    ):
        (
            query_proj_key,
            key_proj_key,
            value_proj_key,
            output_proj_key,
        ) = jax.random.split(key, 4)

        self.query_proj = eqx.nn.Linear(
            num_channels,
            num_channels,
            use_bias=False,
            key=query_proj_key,
        )
        self.key_proj = eqx.nn.Linear(
            num_channels,
            num_channels,
            use_bias=False,
            key=key_proj_key,
        )
        self.value_proj = eqx.nn.Linear(
            num_channels,
            num_channels,
            use_bias=False,
            key=value_proj_key,
        )
        self.output_proj = eqx.nn.Linear(
            num_channels,
            num_channels,
            use_bias=False,
            key=output_proj_key,
        )

        self.num_heads = num_heads

    def __call__(
        self,
        x: Float[Array, "seq_len channels"],
    ) -> Float[Array, "seq_len channels"]:
        seq_len, channels = x.shape

        # Needs to vectorize over the sequence length
        query: Float[Array, "seq_len query_size"] = jax.vmap(self.query_proj)(x)
        key = jax.vmap(self.key_proj)(x)
        value = jax.vmap(self.value_proj)(x)

        # Reshape to split the heads
        query: Float[Array, "seq_len num_heads query_size//num_heads"] = query.reshape(
            (seq_len, self.num_heads, channels // self.num_heads)
        )
        key = key.reshape((seq_len, self.num_heads, channels // self.num_heads))
        value = value.reshape((seq_len, self.num_heads, channels // self.num_heads))

        # Compute attention while vectorizing over the heads
        results = jax.vmap(scaled_dot_product_attention, in_axes=-2, out_axes=-2)(
            query, key, value
        )

        # Concatenate the heads
        results = results.reshape((seq_len, channels))

        # Project the output, vectorizing over the sequence length
        out = jax.vmap(self.output_proj)(results)

        return out


class TransformerBlock(eqx.Module):
    mha: MultiheadAttention
    mlp: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(
        self,
        num_heads: int,
        channels: int,
        mlp_channels_multiplier: int = 2,
        activation: Callable = jax.nn.relu,
        *,
        key: PRNGKeyArray,
    ):
        mha_key, mlp_key = jax.random.split(key, 2)
        self.mha = MultiheadAttention(
            num_heads,
            channels,
            key=mha_key,
        )
        self.mlp = eqx.nn.MLP(
            in_size=channels,
            out_size=channels,
            width_size=mlp_channels_multiplier * channels,
            depth=1,
            activation=activation,
            use_bias=False,
            use_final_bias=False,
            key=mlp_key,
        )
        self.norm1 = eqx.nn.LayerNorm(channels)
        self.norm2 = eqx.nn.LayerNorm(channels)

    def __call__(
        self,
        x: Float[Array, "seq_len channels"],
    ) -> Float[Array, "seq_len channels"]:
        x = jax.vmap(self.norm1)(self.mha(x) + x)
        x = jax.vmap(self.norm2)(jax.vmap(self.mlp)(x) + x)
        return x


def build_positional_encoding_table(
    max_len: int,
    num_channels: int,
    base: int = 10_000,
) -> Float[Array, "max_len num_channels"]:
    pe = jnp.zeros((max_len, num_channels))

    position: Int[Array, "max_len 1"] = jnp.arange(0, max_len)[:, None]
    channel: Int[Array, "1 num_channels"] = jnp.arange(0, num_channels)[None, :]

    even_channel: Int[Array, "1 num_channels/2"] = channel[:, 0::2]

    frequency_scaling: Float[Array, "1 num_channels/2"] = base ** (
        even_channel / num_channels
    )

    pe_even_channels: Float[Array, "max_len num_channels/2"] = jnp.sin(
        position / frequency_scaling
    )
    pe_odd_channels: Float[Array, "max_len num_channels/2"] = jnp.cos(
        position / frequency_scaling
    )

    pe = pe.at[:, 0::2].set(pe_even_channels)
    pe = pe.at[:, 1::2].set(pe_odd_channels)

    return pe


class PositionalEncoding(eqx.Module):
    encoding_table: Float[Array, "max_len channels"]

    def __init__(
        self,
        max_len: int,
        channels: int,
    ):
        self.encoding_table = build_positional_encoding_table(max_len, channels)

    def __call__(
        self,
        x: Float[Array, "seq_len channels"],
    ) -> Float[Array, "seq_len channels"]:
        seq_len = x.shape[0]
        encoding_info = self.encoding_table[:seq_len, :]
        # Avoids backpropagating through the encoding table, it should not be
        # trainable
        encoding_info = jax.lax.stop_gradient(encoding_info)

        return x + encoding_info


class Transformer(eqx.Module):
    blocks: list[TransformerBlock]
    pos_enc: PositionalEncoding

    def __init__(
        self,
        num_heads: int,
        channels: int,
        num_blocks: int,
        mlp_channels_multiplier: int = 2,
        activation: Callable = jax.nn.relu,
        max_length: int = 256,  # maximum length of the sequence
        *,
        key: PRNGKeyArray,
    ):
        blocks = []
        for i in range(num_blocks):
            block_key = jax.random.fold_in(key, i)
            blocks.append(
                TransformerBlock(
                    num_heads=num_heads,
                    channels=channels,
                    mlp_channels_multiplier=mlp_channels_multiplier,
                    activation=activation,
                    key=block_key,
                )
            )
        self.blocks = blocks
        self.pos_enc = PositionalEncoding(max_length, channels)

    def __call__(
        self,
        x: Float[Array, "seq_len channels"],
    ) -> Float[Array, "seq_len channels"]:
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        return x


class TransformerEmulator1d(eqx.Module):
    lifting: eqx.nn.MLP
    transformer: Transformer
    projection: eqx.nn.MLP

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_heads: int,
        num_blocks: int,
        mlp_channels_multiplier: int = 2,
        activation: Callable = jax.nn.relu,
        max_length: int = 256,  # maximum length of the sequence
        *,
        key: PRNGKeyArray,
    ):
        lifting_key, transformer_key, projection_key = jax.random.split(key, 3)

        self.lifting = eqx.nn.MLP(
            in_size=in_channels,
            out_size=hidden_channels,
            width_size=hidden_channels,
            depth=2,  # two hidden layers
            activation=activation,
            use_bias=True,
            use_final_bias=True,
            key=lifting_key,
        )
        self.transformer = Transformer(
            num_heads=num_heads,
            channels=hidden_channels,
            num_blocks=num_blocks,
            mlp_channels_multiplier=mlp_channels_multiplier,
            activation=activation,
            max_length=max_length,
            key=transformer_key,
        )
        self.projection = eqx.nn.MLP(
            in_size=hidden_channels,
            out_size=out_channels,
            width_size=hidden_channels,
            depth=2,  # two hidden layers
            activation=activation,
            use_bias=True,
            use_final_bias=True,
            key=projection_key,
        )

    def __call__(
        self,
        x: Float[Array, "in_channels num_points"],
    ) -> Float[Array, "out_channels num_points"]:
        # Transpose to [num_points in_channels]: num_points is then considered
        # the seq_len
        x = x.T

        # Vectorize over the sequence length (i.e., the num_points)
        x = jax.vmap(self.lifting)(x)

        x = self.transformer(x)

        x = jax.vmap(self.projection)(x)

        # Transpose back into the input format
        return x.T


def transformer_constructor(
    config: str,
    num_spatial_dims: int,
    num_points: int,
    num_channels: int,
    activation_fn,
    key,
):
    if num_spatial_dims != 1:
        raise ValueError("Only 1D supported")
    args = config.split(";")
    hidden_channels = int(args[1])
    num_heads = int(args[2])
    num_blocks = int(args[3])
    max_length = num_points
    return TransformerEmulator1d(
        in_channels=num_channels,
        out_channels=num_channels,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        num_blocks=num_blocks,
        activation=activation_fn,
        max_length=max_length,
        key=key,
    )
