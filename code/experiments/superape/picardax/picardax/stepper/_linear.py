from typing import Literal, Union

import jax.numpy as jnp
import lineax
from jaxtyping import Array, Float

from .._base import CollocatedBaseStepper


class Advection(CollocatedBaseStepper):
    advectivity: tuple[float, ...]

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        advectivity: Union[float, tuple[float, ...]] = 0.1,
        theta: float = 1.0,
        picard_maxiter: int = 100,
        picard_tol: float = 1e-5,
        linsolve: Union[
            lineax.AbstractLinearSolver, Literal["cg", "normalcg", "bicgstab", "gmres"]
        ] = "gmres",
        linsolve_atol: float = 1e-5,
        linsolve_rtol: float = 1e-5,
        linsolve_maxiter: int = None,
        **linsolve_kwargs,
    ):
        if isinstance(advectivity, float):
            advectivity = (advectivity,) * num_spatial_dims
        else:
            if len(advectivity) != num_spatial_dims:
                raise ValueError(
                    f"Length of advectivity must match num_spatial_dims ({num_spatial_dims})"
                )
        self.advectivity = advectivity
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            theta=theta,
            picard_maxiter=picard_maxiter,
            picard_tol=picard_tol,
            linsolve=linsolve,
            linsolve_atol=linsolve_atol,
            linsolve_rtol=linsolve_rtol,
            linsolve_maxiter=linsolve_maxiter,
            **linsolve_kwargs,
        )

    def linearized_derivative_operator(
        self,
        u: Float[Array, "C ... N"],
        u_linearized: Float[Array, "C ... N"],  # unused because linear PDE
    ) -> Float[Array, "C ... N"]:
        winds = jnp.concatenate(
            [
                adv
                * jnp.ones(
                    (1,)
                    + (self.derivatives.num_points,) * self.derivatives.num_spatial_dims
                )
                for adv in self.advectivity
            ]
        )
        return -self.derivatives.scaled_upwind_derivative(u, winds=winds)


class Diffusion(CollocatedBaseStepper):
    diffusivity: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: float = 0.01,
        theta: float = 1.0,
        picard_maxiter: int = 100,
        picard_tol: float = 1e-5,
        linsolve: Union[
            lineax.AbstractLinearSolver, Literal["cg", "normalcg", "bicgstab", "gmres"]
        ] = "gmres",
        linsolve_atol: float = 1e-5,
        linsolve_rtol: float = 1e-5,
        linsolve_maxiter: int = None,
        linsolve_throw: bool = True,
        **linsolve_kwargs,
    ):
        self.diffusivity = diffusivity
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            theta=theta,
            picard_maxiter=picard_maxiter,
            picard_tol=picard_tol,
            linsolve=linsolve,
            linsolve_atol=linsolve_atol,
            linsolve_rtol=linsolve_rtol,
            linsolve_maxiter=linsolve_maxiter,
            linsolve_tags=(lineax.symmetric_tag, lineax.positive_semidefinite_tag),
            linsolve_throw=linsolve_throw,
            **linsolve_kwargs,
        )

    def linearized_derivative_operator(
        self,
        u: Float[Array, "C ... N"],
        u_linearized: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        return self.diffusivity * self.derivatives.laplacian(u)


class Dispersion(CollocatedBaseStepper):
    dispersivity: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        dispersivity: float = 1e-4,
        theta: float = 1.0,
        picard_maxiter: int = 100,
        picard_tol: float = 1e-5,
        linsolve: Union[
            lineax.AbstractLinearSolver, Literal["cg", "normalcg", "bicgstab", "gmres"]
        ] = "gmres",
        linsolve_atol: float = 1e-5,
        linsolve_rtol: float = 1e-5,
        linsolve_maxiter: int = None,
        **linsolve_kwargs,
    ):
        self.dispersivity = dispersivity
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            theta=theta,
            picard_maxiter=picard_maxiter,
            picard_tol=picard_tol,
            linsolve=linsolve,
            linsolve_atol=linsolve_atol,
            linsolve_rtol=linsolve_rtol,
            linsolve_maxiter=linsolve_maxiter,
            **linsolve_kwargs,
        )

    def linearized_derivative_operator(
        self,
        u: Float[Array, "C ... N"],
        u_linearized: Float[Array, "C ... N"],  # unused because linear PDE
    ) -> Float[Array, "C ... N"]:
        winds = self.dispersivity * jnp.ones(
            (self.derivatives.num_spatial_dims,)
            + (self.derivatives.num_points,) * self.derivatives.num_spatial_dims
        )
        # Negative sign to have dissipating effect by default
        return self.derivatives.scaled_upwind_derivative(
            self.derivatives.laplacian(u), winds=winds
        )


class HyperDiffusion(CollocatedBaseStepper):
    hyper_diffusivity: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        hyper_diffusivity: float = 1e-5,
        theta: float = 1.0,
        picard_maxiter: int = 100,
        picard_tol: float = 1e-5,
        linsolve: Union[
            lineax.AbstractLinearSolver, Literal["cg", "normalcg", "bicgstab", "gmres"]
        ] = "gmres",
        linsolve_atol: float = 1e-5,
        linsolve_rtol: float = 1e-5,
        linsolve_maxiter: int = None,
        **linsolve_kwargs,
    ):
        self.hyper_diffusivity = hyper_diffusivity
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            theta=theta,
            picard_maxiter=picard_maxiter,
            picard_tol=picard_tol,
            linsolve=linsolve,
            linsolve_atol=linsolve_atol,
            linsolve_rtol=linsolve_rtol,
            linsolve_maxiter=linsolve_maxiter,
            **linsolve_kwargs,
        )

    def linearized_derivative_operator(
        self,
        u: Float[Array, "C ... N"],
        u_linearized: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        # Negative sign to have dissipating effect by default
        return -self.hyper_diffusivity * self.derivatives.double_laplacian(u)
