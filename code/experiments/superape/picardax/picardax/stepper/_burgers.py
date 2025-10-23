from typing import Literal, Union

import lineax
from jaxtyping import Array, Float

from .._base import CollocatedBaseStepper


class Burgers(CollocatedBaseStepper):
    diffusivity: float
    convection_scale: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: float = 0.1,
        convection_scale: float = 1.0,
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
        self.diffusivity = diffusivity
        self.convection_scale = convection_scale
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
        return -self.convection_scale * self.derivatives.scaled_upwind_derivative(
            u, winds=u_linearized
        ) + self.diffusivity * self.derivatives.laplacian(u)
