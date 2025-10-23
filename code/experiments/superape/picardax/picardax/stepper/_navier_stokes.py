from typing import Literal, Optional, Union

import jax.numpy as jnp
import lineax
from jaxtyping import Array, Float

from .._base import StaggeredBaseStepper


class NavierStokes(StaggeredBaseStepper):
    diffusivity: float
    convection_scale: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: float = 0.001,
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
            non_transient=num_spatial_dims,
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
        vel, p = u[0:2], u[2:3]
        vel_0, vel_1 = vel[0:1], vel[1:2]
        winds = u_linearized[0:2]

        p_grad = self.derivatives.scalar_gradient_on_vel(p)
        vel_div = self.derivatives.velocity_divergence_on_scal(vel)

        vel_0_applied = (
            -self.convection_scale
            * self.derivatives.scaled_upwind_derivative_staggered(
                vel_0,
                winds=winds,
                on=0,
            )
            + self.diffusivity * self.derivatives.laplacian(vel_0)
            - p_grad[0:1]
        )
        vel_1_applied = (
            -self.convection_scale
            * self.derivatives.scaled_upwind_derivative_staggered(
                vel_1,
                winds=winds,
                on=1,
            )
            + self.diffusivity * self.derivatives.laplacian(vel_1)
            - p_grad[1:2]
        )
        p_applied = -vel_div

        return jnp.concatenate(
            [
                vel_0_applied,
                vel_1_applied,
                p_applied,
            ]
        )

    def make_incompressible(
        self,
        u: Float[Array, "C ... N"],
        *,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        maxiter: Optional[int] = None,
    ) -> Float[Array, "C ... N"]:
        vel = u[0:2]
        vel_div = self.derivatives.velocity_divergence_on_scal(vel)

        # TODO: Find out which version to use
        # lin_fun = lambda p: self.derivatives.velocity_divergence(
        #     self.derivatives.scalar_gradient(p)
        # )
        def lin_fun(p):
            p_laplace = self.derivatives.laplacian(p)
            return p_laplace

        lin_operator = lineax.FunctionLinearOperator(lin_fun, vel_div)

        # TODO: this should better be a CG, but I get the error that the linear
        # operator is not spd
        lin_solver_poisson = lineax.GMRES(
            rtol=rtol,
            atol=atol,
            max_steps=maxiter,
        )

        solution_container = lineax.linear_solve(
            lin_operator,
            vel_div,
            lin_solver_poisson,
        )
        p = solution_container.value

        p_grad = self.derivatives.scalar_gradient_on_vel(p)
        vel_updated = vel - p_grad

        return jnp.concatenate(
            [
                vel_updated,
                p,
            ]
        )

    def compute_curl(
        self,
        u: Float[Array, "C ... N"],
    ):
        vel_0, vel_1 = u[0:1], u[1:2]
        d_vel_0_d_1 = self.derivatives.derivative_vel_on_scal(
            vel_0,
            dim=1,
        )
        d_vel_1_d_0 = self.derivatives.derivative_vel_on_scal(
            vel_1,
            dim=0,
        )

        return d_vel_1_d_0 - d_vel_0_d_1
