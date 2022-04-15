"""This module implements the RandomStartGenerator class."""
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


class RandomStartGenerator(MultiStartGenerator):
    """A start generator that selects random points uniformily across [0, 2π)"""

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
            2 * np.pi * np.random.random(circuit.num_params)
            for i in range(multistarts)
        ]
