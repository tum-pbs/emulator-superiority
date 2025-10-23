from . import generic
from ._burgers import Burgers
from ._korteweg_de_vries import KortewegDeVries
from ._kuramoto_sivashinsky import KuramotoSivashinsky, KuramotoSivashinskyConservative
from ._linear import Advection, Diffusion, Dispersion, HyperDiffusion
from ._navier_stokes import NavierStokes

__all__ = [
    "Advection",
    "Diffusion",
    "Dispersion",
    "HyperDiffusion",
    "Burgers",
    "KuramotoSivashinsky",
    "KuramotoSivashinskyConservative",
    "KortewegDeVries",
    "NavierStokes",
    "generic",
]
