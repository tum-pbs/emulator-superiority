import exponax as ex
import lineax
from .picardax import picardax as px


def parse_simulator(
    simulator: str,
    num_points: int,
    linear_difficulties: tuple[float, float, float, float, float],
    convection_difficulty: float,
):
    simulator_name = simulator.split(";")[0].lower()

    if simulator_name == "etdrk":
        order = int(simulator.split(";")[1])
        return ex.stepper.generic.DifficultyConvectionStepper(
            num_spatial_dims=1,
            num_points=num_points,
            linear_difficulties=linear_difficulties,
            convection_difficulty=convection_difficulty,
            order=order,
        )
    elif simulator_name == "fd":
        theta = float(simulator.split(";")[1])
        picard_maxiter = int(simulator.split(";")[2])
        return px.stepper.generic.DifficultyConvection(
            num_spatial_dims=1,
            num_points=num_points,
            diff_linear_coefs=linear_difficulties,
            diff_convection_coef=convection_difficulty,
            theta=theta,
            picard_maxiter=picard_maxiter,
            linsolve=lineax.LU(),
        )
    else:
        raise ValueError(f"Unknown simulator: {simulator}")
