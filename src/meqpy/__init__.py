"""
meqpy: A modular Python framework for defining systems and solving master equations.
"""

__version__ = "0.0.1"

from .system.system import System
from .system.state import State
from .master_equation.equilibrium import (
    solve_equilibrium,
    solve_equilibrium_nd,
    fill_diagonal,
)

__all__ = [
    "System",
    "State",
    "Transition",
    "solve_equilibrium",
    "solve_equilibrium_nd",
    "fill_diagonal",
]
