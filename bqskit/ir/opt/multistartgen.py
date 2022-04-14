"""This module implements the MultiStartGenerator base class."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.state.state import StateVector
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class MultiStartGenerator(abc.ABC):

    @abc.abstractmethod
    def gen_starting_points(
        self,
        multistarts: int,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> list[npt.NDArray[np.float64]]:
        """
        Generate `multistarts` starting points for instantiation.

        Args:
            multistarts (int): The number of starting points to generate.

            circuit (Circuit): The circuit to generate the points for.

            target (UnitaryMatrix | StateVector): The target.

        Return:
            (list[npt.NDArray[np.float64]]): List of starting inputs for
                instantiation.

        Raises:
            ValueError: If `multistarts` is not a positive integer.
        """
