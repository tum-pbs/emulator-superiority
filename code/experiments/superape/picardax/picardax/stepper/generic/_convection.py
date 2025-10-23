from typing import Literal, Union

import jax.numpy as jnp
import lineax
from jaxtyping import Array, Float

from ..._base import CollocatedBaseStepper


class GenericConvection(CollocatedBaseStepper):
    linear_coefs: tuple[float, float, float, float, float]
    convection_coef: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        linear_coefs: tuple[float, float, float, float, float] = (
            0.0,
            0.0,
            0.01,
            0.0,
            0.0,
        ),
        convection_coef: float = -1.0,
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
        """
        All coefficients live on the right-hand side
        """
        self.linear_coefs = linear_coefs
        self.convection_coef = convection_coef
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
        wind_shape = (self.derivatives.num_spatial_dims,) + (
            self.derivatives.num_points,
        ) * self.derivatives.num_spatial_dims
        advection_winds = self.linear_coefs[1] * jnp.ones(wind_shape)
        dispersion_winds = self.linear_coefs[3] * jnp.ones(wind_shape)

        drag_contribution = self.linear_coefs[0] * u
        advection_contribution = self.derivatives.scaled_upwind_derivative(
            -u, winds=-advection_winds
        )
        diffusion_contribution = self.linear_coefs[2] * self.derivatives.laplacian(u)
        dispersion_contribution = self.derivatives.scaled_upwind_derivative(
            self.derivatives.laplacian(u),
            winds=dispersion_winds,
        )
        hyper_diffusion_contribution = self.linear_coefs[
            4
        ] * self.derivatives.double_laplacian(u)

        convection_contribution = (
            self.convection_coef
            * self.derivatives.scaled_upwind_derivative(u, winds=u_linearized)
        )

        return (
            drag_contribution
            + advection_contribution
            + diffusion_contribution
            + dispersion_contribution
            + hyper_diffusion_contribution
            + convection_contribution
        )


class NormalizedConvection(GenericConvection):
    norm_linear_coefs: tuple[float, float, float, float, float]
    norm_convection_coef: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        norm_linear_coefs: tuple[float, float, float, float, float] = (
            0.0,
            0.0,
            0.001,
            0.0,
            0.0,
        ),
        norm_convection_coef: float = -1.0,
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
        """
        All coefficients live on the right-hand side
        """
        self.norm_linear_coefs = norm_linear_coefs
        self.norm_convection_coef = norm_convection_coef

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=1.0,
            num_points=num_points,
            dt=1.0,
            theta=theta,
            linear_coefs=norm_linear_coefs,
            convection_coef=norm_convection_coef,
            picard_maxiter=picard_maxiter,
            picard_tol=picard_tol,
            linsolve=linsolve,
            linsolve_atol=linsolve_atol,
            linsolve_rtol=linsolve_rtol,
            linsolve_maxiter=linsolve_maxiter,
            **linsolve_kwargs,
        )


class DifficultyConvection(NormalizedConvection):
    diff_linear_coefs: tuple[float, float, float, float, float]
    diff_convection_coef: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int = 100,
        *,
        diff_linear_coefs: tuple[float, float, float, float, float] = (
            0.0,
            0.0,
            0.1,
            0.0,
            0.0,
        ),
        diff_convection_coef: float = -1.5,
        maximum_absolute: float = 1.0,
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
        self.diff_linear_coefs = diff_linear_coefs
        self.diff_convection_coef = diff_convection_coef

        norm_linear_coefs = list(
            gamma / (num_points**j * (2 ** (j - 1)) * num_spatial_dims)
            for j, gamma in enumerate(diff_linear_coefs)
        )
        norm_linear_coefs[0] = diff_linear_coefs[0]
        norm_linear_coefs = tuple(norm_linear_coefs)

        norm_convection_coef = diff_convection_coef / (
            maximum_absolute * num_points * num_spatial_dims
        )

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            num_points=num_points,
            norm_linear_coefs=norm_linear_coefs,
            norm_convection_coef=norm_convection_coef,
            theta=theta,
            picard_maxiter=picard_maxiter,
            picard_tol=picard_tol,
            linsolve=linsolve,
            linsolve_atol=linsolve_atol,
            linsolve_rtol=linsolve_rtol,
            linsolve_maxiter=linsolve_maxiter,
            **linsolve_kwargs,
        )
