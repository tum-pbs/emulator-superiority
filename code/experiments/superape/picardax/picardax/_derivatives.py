from typing import Literal, Optional, Sequence, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

stencils = {
    "forward": {
        1: {
            1: [(0, -1), (1, 1)],
        },
        3: {
            1: [(0, -1), (1, 3), (2, -3), (3, 1)],
        },
    },
    "backward": {
        1: {
            1: [(-1, -1), (0, 1)],
        },
        3: {
            1: [(-3, -1), (-2, 3), (-1, -3), (0, 1)],
        },
    },
    "centered": {
        1: {
            2: [(-1, -1 / 2), (1, 1 / 2)],
        },
        2: {
            2: [(-1, 1), (0, -2), (1, 1)],
        },
        3: {
            2: [(-2, -1 / 2), (-1, 1), (1, -1), (2, 1 / 2)],
        },
        4: {
            2: [(-2, 1), (-1, -4), (0, 6), (1, -4), (2, 1)],
        },
    },
}
"""
Finite Difference stencils, extracted from
https://en.wikipedia.org/wiki/Finite_difference_coefficient

Indexing hierarchy is:

1. Finite Difference kind ("forward", "backward", "centered")
2. Derivative Order, e.g. `1` for the first derivative
3. Finite Difference approximition order (=order of consistency)
"""


class CollocatedDerivatives(eqx.Module):
    num_spatial_dims: int
    domain_extent: float
    num_points: int
    dx: float
    indexing: Literal["xy", "ij"]

    @property
    def _derivative_axis(self):
        return -(self.num_spatial_dims + 1)

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        *,
        indexing: Literal["xy", "ij"] = "ij",
    ):
        """
        Take derivatives on collocated variables on a uniform Cartesian grid
        with periodic boundary conditions.

        **Arguments:**

        - `num_spatial_dims`: Number of spatial dimensions, i.e. `1` for 1D, `2`
            for 2D, etc.
        - `domain_extent`: Length of the domain in each spatial dimension.
        - `num_points`: Number of grid points in each spatial dimension,
            **excluding** the redudant right-most grid node.
        - `indexing` (keyword): Indexing convention, either "xy" or "ij"
            following[`numpy.meshgrid`](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html).
            Defaults to "ij" (different from NumPy's default).
        """
        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent
        self.num_points = num_points

        # Possible because of the convention that the right-most grid node is
        # not a degree of freedom.
        self.dx = domain_extent / num_points

        if indexing != "ij":
            raise NotImplementedError("Only 'ij' indexing is supported for now.")
        self.indexing = indexing

    def get_neighbor(
        self,
        u: Float[Array, "C ... N"],
        *,
        dim: Union[int, Sequence[int]],
        shift: Union[int, Sequence[int]],
    ) -> Float[Array, "C ... N"]:
        """
        Shift variables in a given dimension by a given amount. Used to
        implement finite difference schemes for derivative approximation. Due to
        periodic boundary conditions, rolls across the array bounds.

        **Arguments:**

        - `u`: The field to shift.
        - `dim` (keyword): The dimension(s) to shift.
        - `shift` (keyword): The amount(s) to shift.

        **Returns:**

        - The shifted field, has the same shape as the input field

        !!! info
            On staggered grids, this only works if one stays within one grid.
        """
        if isinstance(dim, int):
            dim = (dim,)
        if isinstance(shift, int):
            shift = (shift,)
        axis_to_roll = (-self.num_spatial_dims + d for d in dim)
        # Needs the minus sign because rolling follows the "MacOS" convention
        roll = (-s for s in shift)
        return jnp.roll(u, roll, axis=axis_to_roll)

    def derivative(
        self,
        u: Float[Array, "C ... N"],
        *,
        dim: int,
        derivative_order: int,
        method: Literal["forward", "backward", "centered"],
        method_order: int,
    ) -> Float[Array, "C ... N"]:
        """
        Compute a derivative on a field based on a finite difference
        approximation.

        **Arguments:**

        - `u`: The field to differentiate.
        - `dim`: The dimension to differentiate along.
        - `derivative_order`: The order of the derivative, e.g. `1` for the
            first derivative, `2` for the second derivative, etc.
        - `method`: The finite difference method to use, either "forward",
            "backward", or "centered".
        - `method_order`: The order of the finite difference method, e.g. `1`
            for a first-order approximation, `2` for a second-order
            approximation, etc.

        **Returns:**

        - The derivative of the field, has the same shape as the input field

        !!! info

            Some derivative orders are not available in all approximation
            orders. The coefficients are taken from
            https://en.wikipedia.org/wiki/Finite_difference_coefficient
        """
        rolls_and_weights = stencils[method][derivative_order][method_order]

        return (
            sum(
                weight * self.get_neighbor(u, dim=dim, shift=shift)
                for shift, weight in rolls_and_weights
            )
            / (self.dx) ** derivative_order
        )

    def get_positive_wind(
        self,
        winds: Float[Array, "D ... N"],
        *,
        dim: Optional[int] = None,
        zero_fixed: bool = True,
    ) -> Float[Array, "D ... N"]:
        """
        Only keep the entries with positive value, set the rest to zero.

        **Arguments:**

        - `winds`: The wind field.
        - `dim` (keyword): The dimension to consider, only needed if
            `zero_fixed=True` (default).
        - `zero_fixed` (keyword): Whether to employ a fix that winds which are
            next to non-zero regions are also non-zero. For simulating the
            Burgers equation, this is crucial to have shocks with opposing signs
            propagate correctly.


        **Returns:**

        - The positive wind field with the same shape as the input.
        """
        if zero_fixed:
            if dim is None:
                raise ValueError("dim must be provided if zero_fixed=True.")
            return jnp.maximum(
                (winds + self.get_neighbor(winds, dim=dim, shift=-1)) / 2,
                0.0,
            )
        else:
            return jnp.maximum(winds, 0.0)

    def get_negative_wind(
        self,
        winds: Float[Array, "D ... N"],
        *,
        dim: Optional[int] = None,
        zero_fixed: bool = True,
    ) -> Float[Array, "D ... N"]:
        """
        Only keep the entries with negative value, set the rest to zero.

        **Arguments:**

        - `winds`: The wind field.
        - `dim` (keyword): The dimension to consider, only needed if
            `zero_fixed=True` (default).
        - `zero_fixed` (keyword): Whether to employ a fix that winds which are
            next to non-zero regions are also non-zero. For simulating the
            Burgers equation, this is crucial to have shocks with opposing signs
            propagate correctly.


        **Returns:**

        - The negative wind field with the same shape as the input.
        """
        if zero_fixed:
            if dim is None:
                raise ValueError("dim must be provided if zero_fixed=True.")
            return jnp.minimum(
                (winds + self.get_neighbor(winds, dim=dim, shift=1)) / 2,
                0.0,
            )
        else:
            return jnp.minimum(winds, 0.0)

    def gradient(
        self,
        u: Float[Array, "C ... N"],
        *,
        derivative_order: int = 1,
        method: Literal["forward", "backward", "centered"],
        method_order: int,
    ) -> Float[Array, "C D ... N"]:
        """
        Collection of all partial derivatives, adds an additional derivative
        axis to the field.

        **Arguments:**

        - `u`: The field to differentiate.
        - `derivative_order`: The order of the derivative, e.g. `1` for the
            first derivative, `2` for the second derivative, etc. In symbolic
            notation, this behaves as:

            - `derivative_order = 2`: `(∇ ⊙ ∇) u`
            - `derivative_order = 3`: `(∇ ⊙ ∇ ⊙ ∇) u`
            - etc.

            (i.e., no cross-derivatives are included)
        - `method`: The finite difference method to use, either "forward",
            "backward", or "centered".
        - `method_order`: The order of the finite difference method, e.g. `1`
            for a first-order approximation, `2` for a second-order
            approximation, etc. Not all derivative methods and orders support
            all approximation orders.

        **Returns:**

        - The gradient of the field, has one additional axis with as many
            dimensions as `self.num_spatial_dims` position right before the
            spatial axes in the end of the array shape.
        """
        return jnp.stack(
            [
                self.derivative(
                    u,
                    dim=dim,
                    derivative_order=derivative_order,
                    method=method,
                    method_order=method_order,
                )
                for dim in range(self.num_spatial_dims)
            ],
            axis=self._derivative_axis,
        )

    def laplacian(
        self,
        u: Float[Array, "C ... N"],
        *,
        method: Literal["forward", "backward", "centered"] = "centered",
        method_order: int = 2,
    ) -> Float[Array, "C ... N"]:
        """
        Sum of all second-order partial derivatives.

        **Arguments:**

        - `u`: The field to differentiate.
        - `method`: The finite difference method to use, either "forward",
            "backward", or "centered".
        - `method_order`: The order of the finite difference method, e.g. `1`


        **Returns:**

        - The Laplacian of the field, has the same shape as the input field
        """
        return jnp.sum(
            self.gradient(
                u,
                derivative_order=2,
                method=method,
                method_order=method_order,
            ),
            axis=self._derivative_axis,
        )

    def double_laplacian(
        self,
        u: Float[Array, "C ... N"],
        *,
        mixed: bool = False,
        method: Literal["forward", "backward", "centered"] = "centered",
        method_order: int = 2,
    ):
        """
        The double laplacian applied to a field

        Either in a spatially mixed form, with

        `Δ Δ u = (∇ ⋅ ∇) (∇ ⋅ ∇) u`

        or in a form without cross derivatives (default), with

        `1 ⋅ (∇ ⊙ ∇ ⊙ ∇ ⊙ ∇) u`

        **Arguments:**

        - `u`: The field to differentiate.
        - `mixed` (keyword): Whether to include cross derivatives.
        - `method`: The finite difference method to use, either "forward",
            "backward", or "centered".
        - `method_order`: The order of the finite difference method, e.g. `1`
            for a first-order approximation, `2` for a second-order
            approximation, etc.


        **Returns:**

        - The double Laplacian of the field, has the same shape as the input field
        """
        if mixed:
            raise NotImplementedError("mixed=True not implemented yet")

        return jnp.sum(
            self.gradient(
                u,
                derivative_order=4,
                method=method,
                method_order=method_order,
            ),
            axis=self._derivative_axis,
        )

    def gradient_norm(
        self,
        u: Float[Array, "C ... N"],
        *,
        derivative_order: int = 1,
        method: Literal["forward", "backward", "centered"],
        method_order: int,
    ) -> Float[Array, "C ... N"]:
        """
        Compute the norm of the gradient. ‖∇ u‖₂

        !!! warning

            Caution: This is a nonlinear operation!


        **Arguments:**

        - `u`: The field to differentiate.
        - `derivative_order`: The order of the derivative, e.g. `1` for the
            first derivative, `2` for the second derivative, etc. See
            [`picardax.CollocatedDerivatives.gradient`][] for more information.
        - `method`: The finite difference method to use, either "forward",
            "backward", or "centered".
        - `method_order`: The order of the finite difference method, e.g. `1`
            for a first-order approximation, `2` for a second-order
            approximation, etc.


        **Returns:**

        - The norm of the gradient of the field, has the same shape as the input
            field
        """
        return jnp.linalg.norm(
            self.gradient(
                u,
                derivative_order=derivative_order,
                method=method,
                method_order=method_order,
            ),
            axis=self._derivative_axis,
        )

    def scaled_upwind_derivative(
        self,
        u: Float[Array, "C ... N"],
        *,
        winds: Float[Array, "D ... N"],
        zero_fixed: bool = True,
    ) -> Float[Array, "C ... N"]:
        """
        Compute the upwind derivative of a field where both `u` and `winds` are
        on the same collocated grid.

        It computes `(w ⋅ ∇)u` where `w` is the wind field.

        **Arguments:**

        - `u`: The field to differentiate.
        - `winds`: The wind field.
        - `zero_fixed` (keyword): Whether to employ a fix that winds which are
            next to non-zero regions are also non-zero. For simulating the
            Burgers equation, this is crucial to have shocks with opposing signs
            propagate correctly.

        **Returns:**

        - The upwind derivative of the field, has the same shape as the input
            field.
        """
        positive_winds: Float[Array, "1 D ... N"] = jnp.stack(
            [
                self.get_positive_wind(
                    winds[dim : dim + 1],
                    dim=dim,
                    zero_fixed=zero_fixed,
                )
                for dim in range(self.num_spatial_dims)
            ],
            axis=self._derivative_axis,
        )
        negative_winds: Float[Array, "1 D ... N"] = jnp.stack(
            [
                self.get_negative_wind(
                    winds[dim : dim + 1],
                    dim=dim,
                    zero_fixed=zero_fixed,
                )
                for dim in range(self.num_spatial_dims)
            ],
            axis=self._derivative_axis,
        )

        gradient_forward = self.gradient(
            u,
            method="forward",
            method_order=1,
        )
        gradient_backward = self.gradient(
            u,
            method="backward",
            method_order=1,
        )

        upwind_derivative = jnp.sum(
            (positive_winds * gradient_backward + negative_winds * gradient_forward),
            axis=self._derivative_axis,
        )

        return upwind_derivative


class StaggeredDerivatives(CollocatedDerivatives):
    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        *,
        indexing: Literal["xy", "ij"] = "ij",
    ):
        """
        Take derivatives on staggered variables on a uniform Cartesian grid with
        periodic boundary conditions.

        **Arguments:**

        - `num_spatial_dims`: Number of spatial dimensions, i.e. `1` for 1D, `2`
            for 2D, etc.
        - `domain_extent`: Length of the domain in each spatial dimension.
        - `num_points`: Number of grid points in each spatial dimension,
            **excluding** the redudant right-most grid node.
        - `indexing` (keyword): Indexing convention, either "xy" or "ij"
            following[`numpy.meshgrid`](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html).
            Defaults to "ij" (different from NumPy's default).
        """
        if num_spatial_dims != 2:
            raise ValueError("Only 2D is supported for staggered grids.")
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            indexing=indexing,
        )

    def _map_vel_0_to_vel_1_grid(
        self, vel_0: Float[Array, "1 ... N"]
    ) -> Float[Array, "1 ... N"]:
        return (
            self.get_neighbor(vel_0, dim=0, shift=-1)
            + vel_0
            + self.get_neighbor(vel_0, dim=(0, 1), shift=(-1, 1))
            + self.get_neighbor(vel_0, dim=1, shift=1)
        ) / 4

    def _map_vel_1_to_vel_0_grid(
        self, vel_1: Float[Array, "1 ... N"]
    ) -> Float[Array, "1 ... N"]:
        return (
            self.get_neighbor(vel_1, dim=1, shift=-1)
            + self.get_neighbor(vel_1, dim=(0, 1), shift=(1, -1))
            + vel_1
            + self.get_neighbor(vel_1, dim=0, shift=1)
        ) / 4

    def map_vel(
        self, vel: Float[Array, "1 ... N"], *, origin: int, destination: int
    ) -> Float[Array, "1 ... N"]:
        if origin == 0:
            if destination == 0:
                return vel
            elif destination == 1:
                return self._map_vel_0_to_vel_1_grid(vel)
            else:
                raise ValueError("Invalid destination.")
        elif origin == 1:
            if destination == 0:
                return self._map_vel_1_to_vel_0_grid(vel)
            elif destination == 1:
                return vel
            else:
                raise ValueError("Invalid destination.")

    def derivative_vel_on_scal(
        self,
        vel: Float[Array, "1 ... N"],
        *,
        dim: int,
        derivative_order: int = 1,
        method: Literal["centered"] = "centered",
    ) -> Float[Array, "1 ... N"]:
        """
        Compute a derivative of field on staggered representation to then be on
        the centered representation.

        This can be used to take the divergence of a velocity field to build the
        right-hand side of a pressure Poisson problem.

        **Arguments:**

        - `vel`: The field to differentiate.
        - `dim`: The dimension to differentiate along.
        - `derivative_order`: The order of the derivative, e.g. `1` for the
            first derivative, `2` for the second derivative, etc.
        - `method`: The finite difference method to use, either "forward",
            "backward", or "centered".

        **Returns:**

        - The derivative of the field, has the same shape as the input field
        """
        if method == "centered":
            if derivative_order == 1:
                return self.derivative(
                    vel,
                    dim=dim,
                    derivative_order=1,
                    # backward derivative on a forward staggered grid is the
                    # second-order centrered derivative for the velocities if
                    # thought on the scalar grid.
                    method="backward",
                    method_order=1,
                )
            else:
                raise ValueError("Only first derivative for staggered is supported.")
        else:
            raise ValueError("Invalid method.")

    def derivative_scal_on_vel(
        self,
        scal: Float[Array, "1 ... N"],
        *,
        dim: int,
        derivative_order: int = 1,
        method: Literal["centered"] = "centered",
    ) -> Float[Array, "1 ... N"]:
        """
        Compute a derivative of field on centered representation to then be on
        the staggered representation.

        This can be used to take the gradient of the pressure field to correct
        velocities to be incompressible.

        **Arguments:**

        - `scal`: The field to differentiate.
        - `dim`: The dimension to differentiate along.
        - `derivative_order`: The order of the derivative, e.g. `1` for the
            first derivative, `2` for the second derivative, etc.
        - `method`: The finite difference method to use, either "forward",
            "backward", or "centered".


        **Returns:**

        - The derivative of the field, has the same shape as the input field
        """
        if method == "centered":
            if derivative_order == 1:
                return self.derivative(
                    scal,
                    dim=dim,
                    derivative_order=1,
                    # forward derivative on a forward staggered grid is the
                    # second-order centrered derivative for the scalars if
                    # thought on the velocity grid.
                    method="forward",
                    method_order=1,
                )
            else:
                raise ValueError("Only first derivative for staggered is supported.")
        else:
            raise ValueError("Invalid method.")

    def scalar_gradient_on_vel(
        self,
        scal: Float[Array, "C ... N"],
        *,
        only_channel_zero: bool = True,
    ) -> Union[Float[Array, "D ... N"], Float[Array, "C D ... N"]]:
        """
        Compute the gradient of a scalar field on the staggered grid.

        **Arguments:**

        - `scal`: The field to differentiate.
        - `only_channel_zero` (keyword): Whether to only return the gradient
            only for the first channel. Set to `True` by default because the
            pressure field is a scalar field.

        **Returns:**

        - The gradient of the field, has one additional axis with as many
            dimensions as `self.num_spatial_dims` position right before the
            spatial axes in the end of the array shape. If `only_channel_zero`
            is `True`, the gradient is only computed for the first/zeroth
            channel.
        """
        grad = jnp.stack(
            [
                self.derivative_scal_on_vel(scal, dim=dim)
                for dim in range(self.num_spatial_dims)
            ],
            axis=self._derivative_axis,
        )
        if only_channel_zero:
            return grad[0]
        else:
            return grad

    def velocity_divergence_on_scal(
        self,
        vel: Float[Array, "D ... N"],
    ) -> Float[Array, "1 ... N"]:
        """
        Compute the divergence of a velocity field on the centered grid.

        **Arguments:**

        - `vel`: The field to differentiate.

        **Returns:**

        - The divergence of the field, has the same shape as the input field
        """
        return sum(
            self.derivative_vel_on_scal(vel[dim : dim + 1], dim=dim)
            for dim in range(self.num_spatial_dims)
        )

    def scaled_upwind_derivative_staggered(
        self,
        field: Float[Array, "C ... N"],
        *,
        on: int,
        winds: Float[Array, "D ... N"],
        zero_fixed: bool = True,
    ) -> Float[Array, "C ... N"]:
        """
        Compute the upwind derivative of a field where both `field` and `winds`
        are on staggered representation.

        It computes `(w ⋅ ∇)u` where `w` is the wind field.

        **Arguments:**

        - `field`: The field to differentiate.
        - `on`: The dimension of the staggered representation, i.e., whether it
            is on the grid for velocity componentn 0 (-> 0) or 1 (-> 1).
        - `winds`: The wind field.
        - `zero_fixed` (keyword): Whether to employ a fix that winds which are
            next to non-zero regions are also non-zero. For simulating the
            Burgers equation, this is crucial to have shocks with opposing signs
            propagate correctly.

        **Returns:**

        - The upwind derivative of the field, has the same shape as the input
            field.
        """
        winds_mapped = jnp.concatenate(
            [
                self.map_vel(winds[dim : dim + 1], origin=dim, destination=on)
                for dim in range(self.num_spatial_dims)
            ]
        )
        return self.scaled_upwind_derivative(
            field,
            winds=winds_mapped,
            zero_fixed=zero_fixed,
        )
