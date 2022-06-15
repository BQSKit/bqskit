"""This module implements the DiagonalStartGenerator base class."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bqskit.ir.opt.multistartgen import MultiStartGenerator
from bqskit.utils.typing import is_integer

if TYPE_CHECKING:
    import numpy.typing as npt
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.state.state import StateVector
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class DiagonalStartGenerator(MultiStartGenerator):
    """A generator that puts starts along the diagonal of the N-d space."""

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

        if not is_integer(multistarts):
            raise TypeError(
                'Expected int for multistarts, got %s.' % type(multistarts),
            )

        if multistarts <= 0:
            raise ValueError(
                'Expected positive integer for multistarts'
                ', got %d' % multistarts,
            )

        return [
            2 * np.pi * np.random.uniform(
                (i - 1) / multistarts,
                i / multistarts,
                (circuit.num_params,),
            )
            for i in range(1, multistarts + 1)
        ]
