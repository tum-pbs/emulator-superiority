from typing import Callable, Literal, Optional, Sequence, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxopt
import lineax
from jaxtyping import Array, Float

from ._derivatives import CollocatedDerivatives, StaggeredDerivatives


def parse_linsolve(
    linsolve: Literal[
        "cg",
        "normalcg",
        "bicgstab",
        "gmres",
    ],
    *,
    linsolve_rtol: float,
    linsolve_atol: float,
    linsolve_maxiter: int,
    **linsolve_kwargs,
):
    if linsolve == "cg":
        return lineax.CG(
            rtol=linsolve_rtol,
            atol=linsolve_atol,
            max_steps=linsolve_maxiter,
            **linsolve_kwargs,
        )
    elif linsolve == "normalcg":
        return lineax.NormalCG(
            rtol=linsolve_rtol,
            atol=linsolve_atol,
            max_steps=linsolve_maxiter,
            **linsolve_kwargs,
        )
    elif linsolve == "bicgstab":
        return lineax.BiCGStab(
            rtol=linsolve_rtol,
            atol=linsolve_atol,
            max_steps=linsolve_maxiter,
            **linsolve_kwargs,
        )
    elif linsolve == "gmres":
        return lineax.GMRES(
            rtol=linsolve_rtol,
            atol=linsolve_atol,
            max_steps=linsolve_maxiter,
            **linsolve_kwargs,
        )
    else:
        raise ValueError(f"Unknown linear solver: {linsolve}")


class PicardStepper(eqx.Module):
    picard_maxiter: int
    picard_tol: float
    linsolve: lineax.AbstractLinearSolver
    linsolve_tags: tuple
    linsolve_throw: bool

    def __init__(
        self,
        *,
        picard_maxiter: int = 100,
        picard_tol: float = 1e-5,
        linsolve: Union[
            lineax.AbstractLinearSolver, Literal["cg", "normalcg", "bicgstab", "gmres"]
        ] = "gmres",
        linsolve_atol: float = 1e-5,
        linsolve_rtol: float = 1e-5,
        linsolve_maxiter: int = None,
        linsolve_tags: tuple = (),
        linsolve_throw: bool = True,
        **linsolve_kwargs,
    ):
        self.picard_maxiter = picard_maxiter
        self.picard_tol = picard_tol

        if isinstance(linsolve, lineax.AbstractLinearSolver):
            self.linsolve = linsolve
        else:
            self.linsolve = parse_linsolve(
                linsolve,
                linsolve_rtol=linsolve_rtol,
                linsolve_atol=linsolve_atol,
                linsolve_maxiter=linsolve_maxiter,
                **linsolve_kwargs,
            )

        self.linsolve_tags = linsolve_tags
        self.linsolve_throw = linsolve_throw

    def linearized_residuum(
        self,
        u_next: Float[Array, "C ... N"],
        u_prev: Float[Array, "C ... N"],
        u_linearized: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        raise NotImplementedError("Subclasses must implement this method")

    def residuum(
        self, u_next: Float[Array, "C ... N"], u_prev: Float[Array, "C ... N"]
    ) -> Float[Array, "C ... N"]:
        return self.linearized_residuum(u_next, u_prev, u_next)

    def assmble_rhs(
        self,
        u_prev: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        """
        Extract the right hand side of the linearized residuum. Given the
        _linearized_residuum is of the form

            g(u, U, v) = A(U)u - b(v)

        we get the rhs by

            b(v) = -g(0, ⋅, v)

        In other words, we set u = 0 and negate the result. The value of the
        linearization point U does not matter, because the linearized residuum
        is linear in u and hence multiplied by 0.

        We set it U = 0 for convenience, because it requires the input to be of
        a certain shape.
        """
        neg_rhs = self.linearized_residuum(
            jnp.zeros_like(u_prev),
            u_prev,
            jnp.zeros_like(u_prev),
        )
        rhs = -neg_rhs
        return rhs

    def assemble_lin_fun(
        self, u_linearized: Float[Array, "C ... N"]
    ) -> Callable[[Float[Array, "C ... N"]], Float[Array, "C ... N"]]:
        """
        Returns a linear function that applies the effect of A(U) to a given
        vector u. The linearization point U is fixed.

        Given the _linearized_residuum is of the form

            g(u, U, v) = A(U)u - b(v)

        we get the linear function by

            f(u) = A(U)u = g(u, U, 0)

        In other words, we set v = 0. The value of the linearization point is
        captured in the closure.

        Args:
            - `linearize_at`: The linearization point U.

        Returns:
            A linear function that applies the effect of A(U) to a given vector
            u. Has the signature `f(u: Array) -> Array`.
        """
        lin_fun = lambda u: self.linearized_residuum(
            u,
            jnp.zeros_like(u_linearized),
            u_linearized,
        )
        return lin_fun

    def assemble_lin_operator(
        self, linearize_at: Float[Array, "C ... N"]
    ) -> lineax.FunctionLinearOperator:
        """
        Same as `assemble_lin_fun`, but returns a `lineax.FunctionLinearOperator`
        instead.
        """
        lin_fun = self.assemble_lin_fun(linearize_at)
        lin_operator = lineax.FunctionLinearOperator(
            lin_fun,
            linearize_at,
            tags=self.linsolve_tags,
        )
        return lin_operator

    def materialize_system_matrix(
        self,
        linearize_at: Float[Array, "C ... N"],
    ) -> Float[Array, "X X"]:
        """
        Can be large and is materialized including its zeros!

        Can be used to inspect the sparsity pattern of the system matrix (e.g.
        via `plt.spy)
        """
        lin_operator = self.assemble_lin_operator(linearize_at)
        system_matrix = lin_operator.as_matrix()
        return system_matrix

    def materialze_rhs_assembly_matrix(
        self,
        u_prev: Float[Array, "C ... N"],
    ):
        """
        Matrix is only constant if the method is fully implicit (all
        nonlinearities are treated implicitly) or if the PDE is linear.

        Returns the Jacobian of the right hand side assembly.
        """
        rhs_matrix = jax.jacfwd(
            lambda u: self.assmble_rhs(u.reshape(u_prev.shape)).flatten()
        )(u_prev.flatten())
        return rhs_matrix

    def _picard_step(
        self,
        u_current_iter: Float[Array, "C ... N"],
        rhs: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        """
        Perform one iteration of the Picard method.

        This consists of two steps:

        1. Re-Assemble the system matrix at the current iterate. (Here we will
           produce a new linear operator that applies the effect of the matrix)
        2. Solve the linear system at the rhs (this rhs is constant for all
           Picard iterations and contains the information of the previous time
           step)

        Args:
            - `u_current_iter`: The current iterate.
            - `rhs`: The right hand side of the linear system. This is constant
                for all Picard iterations.

        Returns:
            The next iterate.
        """
        lin_operator = self.assemble_lin_operator(u_current_iter)
        solution_container = self.perform_linsolve(lin_operator, rhs)
        u_next_iter = solution_container.value
        return u_next_iter

    def p1_step(
        self,
        u_prev: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        rhs = self.assmble_rhs(u_prev)
        return self._picard_step(u_prev, rhs)

    def get_initial_guess(
        self,
        u_prev: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        """
        Give initial guess for Picard iteration, perform necessary autodiff
        detachment if relevant.
        """
        initial_guess = jax.lax.stop_gradient(u_prev)
        return initial_guess

    def _run_iterator(
        self,
        u_prev: Float[Array, "C ... N"],
    ) -> jaxopt.OptStep:
        initial_guess = self.get_initial_guess(u_prev)
        rhs = self.assmble_rhs(u_prev)

        iterator = jaxopt.FixedPointIteration(
            fixed_point_fun=self._picard_step,
            maxiter=self.picard_maxiter,
            tol=self.picard_tol,
        )

        res = iterator.run(
            initial_guess,
            rhs,
        )
        return res

    def _picard_solve(
        self,
        u_prev: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        """
        Perform a Picard iteration until convergence or maxiter iterations have
        been performed; whatever comes first.

        The initial guess is the previous solution.

        Args:
            - `u_prev`: The previous solution.

        Returns:
            The next solution.
        """
        res = self._run_iterator(u_prev)
        u_next = res.params
        return u_next

    def diagnose_picard(
        self,
        u_prev: Float[Array, "C ... N"],
    ) -> int:
        """
        Returns how many iterations it took the Picard solver until the
        nonlinear residuum was reduced to the desired tolerance (convergence was
        achieved).

        Runs at max to `self.picard_maxiter`
        """
        res = self._run_iterator(u_prev)
        num_iterations_needed = res.state.iter_num
        return num_iterations_needed

    def perform_linsolve(
        self,
        lin_operator: lineax.AbstractLinearOperator,
        rhs: Float[Array, "C ... N"],
    ):
        solution_container = lineax.linear_solve(
            lin_operator,
            rhs,
            self.linsolve,
            throw=self.linsolve_throw,
        )
        return solution_container

    def diagnose_linsolve(
        self,
        u_current_iter: Float[Array, "C ... N"],
        u_prev: Float[Array, "C ... N"],
    ) -> int:
        """
        Return number of iteration linear solver takes to convergence at one
        specific Picard iteration. Only works for iterative linear solvers from
        Lineax.
        """
        lin_operator = self.assemble_lin_operator(u_current_iter)
        rhs = self.assmble_rhs(u_prev)
        solution_container = self.perform_linsolve(lin_operator, rhs)

        num_iterations_needed = solution_container.stats["num_steps"]

        return num_iterations_needed

    def get_all_iterates(
        self,
        u_prev: Array,
        *,
        include_init: bool = False,
    ) -> Array:
        """
        Returns all iterates of the Picard iteration.

        Args:
            - `u_prev`: The previous solution.
            - `include_init`: Whether to include the initial guess in the
                returned array. By default, the initial guess is the previous
                solution. Default: False.

        Returns:
            An array of shape (n_iterates, *u_prev.shape) containing all
            iterates of the Picard iteration. `n_iterates` is `self.maxiter`
            even if the iteration converged earlier. Beyond convergence, the
            iterates are not meaningful anymore.
        """

        rhs = self.assmble_rhs(u_prev)

        def scan_fn(u_current_iter, _):
            u_next_iter = self._picard_step(u_current_iter, rhs)
            return u_next_iter, u_next_iter

        initial_guess = self.get_initial_guess(u_prev)

        _, all_iterates = jax.lax.scan(
            scan_fn,
            initial_guess,
            None,
            length=self.picard_maxiter,
        )

        if include_init:
            all_iterates = jnp.concatenate(
                [jnp.expand_dims(initial_guess, axis=0), all_iterates], axis=0
            )

        return all_iterates

    def __call__(
        self,
        u_prev: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        """
        Perform a Picard iteration until convergence or maxiter iterations have
        been performed; whatever comes first.

        The initial guess is the previous solution.

        Args:
            - `u_prev`: The previous solution.

        Returns:
            The next solution.
        """
        u_next = self._picard_solve(u_prev)
        return u_next


class ThetaTimeStepper(PicardStepper):
    dt: float
    theta: float
    """Must be in the range [0, 1].

    - 0: Explicit Euler
    - 1: Implicit Euler
    - 0.5: Crank-Nicolson
    """
    non_transient: Optional[Union[int, Sequence[int]]]

    def __init__(
        self,
        dt: float,
        *,
        theta: float = 1.0,
        non_transient: Optional[Union[int, Sequence[int]]] = None,
        picard_maxiter: int = 100,
        picard_tol: float = 1e-5,
        linsolve: Union[
            lineax.AbstractLinearSolver, Literal["cg", "normalcg", "bicgstab", "gmres"]
        ] = "gmres",
        linsolve_atol: float = 1e-5,
        linsolve_rtol: float = 1e-5,
        linsolve_maxiter: int = None,
        linsolve_tags: tuple = (),
        linsolve_throw: bool = True,
        **linsolve_kwargs,
    ):
        """
        non_transient: channel indices specifying PDEs without first-order derivative
        """
        self.dt = dt
        self.theta = theta
        self.non_transient = non_transient
        super().__init__(
            picard_maxiter=picard_maxiter,
            picard_tol=picard_tol,
            linsolve=linsolve,
            linsolve_atol=linsolve_atol,
            linsolve_rtol=linsolve_rtol,
            linsolve_maxiter=linsolve_maxiter,
            linsolve_tags=linsolve_tags,
            linsolve_throw=linsolve_throw,
            **linsolve_kwargs,
        )

    def linearized_derivative_operator(
        self,
        u: Float[Array, "C ... N"],
        u_linearized: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        raise NotImplementedError("Subclasses must implement this method")

    def linearized_residuum(
        self,
        u_next: Float[Array, "C ... N"],
        u_prev: Float[Array, "C ... N"],
        u_linearized: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        u_next_contribution = self.linearized_derivative_operator(
            u_next,
            u_linearized,
        )

        u_prev_contribution = self.linearized_derivative_operator(
            u_prev,
            u_prev,
        )

        non_transient_residual = (
            self.theta * u_next_contribution + (1 - self.theta) * u_prev_contribution
        )
        residual = (
            u_next
            - u_prev
            - self.dt * non_transient_residual  # Minus because moved to the lhs
        )

        # residual = (
        #     u_next
        #     - u_prev
        #     - self.dt  # Minus because moved to the lhs
        #     * (
        #         self.theta * u_next_contribution
        #         + (1 - self.theta) * u_prev_contribution
        #     )
        # )

        if self.non_transient is not None:
            residual = residual.at[self.non_transient].set(
                non_transient_residual[self.non_transient]
            )

        return residual


class CollocatedBaseStepper(ThetaTimeStepper):
    derivatives: CollocatedDerivatives

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        theta: float = 1.0,
        non_transient: Optional[Union[int, Sequence[int]]] = None,
        picard_maxiter: int = 100,
        picard_tol: float = 1e-5,
        linsolve: Union[
            lineax.AbstractLinearSolver, Literal["cg", "normalcg", "bicgstab", "gmres"]
        ] = "gmres",
        linsolve_atol: float = 1e-5,
        linsolve_rtol: float = 1e-5,
        linsolve_maxiter: int = None,
        linsolve_tags: tuple = (),
        linsolve_throw: bool = True,
        **linsolve_kwargs,
    ):
        self.derivatives = CollocatedDerivatives(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
        )
        super().__init__(
            dt=dt,
            theta=theta,
            non_transient=non_transient,
            picard_maxiter=picard_maxiter,
            picard_tol=picard_tol,
            linsolve=linsolve,
            linsolve_atol=linsolve_atol,
            linsolve_rtol=linsolve_rtol,
            linsolve_maxiter=linsolve_maxiter,
            linsolve_tags=linsolve_tags,
            linsolve_throw=linsolve_throw,
            **linsolve_kwargs,
        )


class StaggeredBaseStepper(ThetaTimeStepper):
    derivatives: StaggeredDerivatives

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        theta: float = 1.0,
        non_transient: Optional[Union[int, Sequence[int]]] = None,
        picard_maxiter: int = 100,
        picard_tol: float = 1e-5,
        linsolve: Union[
            lineax.AbstractLinearSolver, Literal["cg", "normalcg", "bicgstab", "gmres"]
        ] = "gmres",
        linsolve_atol: float = 1e-5,
        linsolve_rtol: float = 1e-5,
        linsolve_maxiter: int = None,
        linsolve_tags: tuple = (),
        linsolve_throw: bool = True,
        **linsolve_kwargs,
    ):
        self.derivatives = StaggeredDerivatives(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
        )
        super().__init__(
            dt=dt,
            theta=theta,
            non_transient=non_transient,
            picard_maxiter=picard_maxiter,
            picard_tol=picard_tol,
            linsolve=linsolve,
            linsolve_atol=linsolve_atol,
            linsolve_rtol=linsolve_rtol,
            linsolve_maxiter=linsolve_maxiter,
            linsolve_tags=linsolve_tags,
            linsolve_throw=linsolve_throw,
            **linsolve_kwargs,
        )
