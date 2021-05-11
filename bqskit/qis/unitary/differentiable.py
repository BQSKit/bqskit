"""This module implements the DifferentiableUnitary base class."""
from __future__ import annotations

import abc
from typing import Sequence

import numpy as np

from bqskit.qis.unitary.unitary import Unitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class DifferentiableUnitary(Unitary):
    """
    The DifferentiableUnitary base class.

    A DifferentiableUnitary exposes the `get_grad` abstract method and the
    `get_unitary_and_grad` method.
    """

    @abc.abstractmethod
    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """
        Return the gradient for the unitary as a np.ndarray.

        Args:
            params (Sequence[float]): The unitary parameters, see
                `Unitary.get_unitary` for more info.

        Returns:
            (np.ndarray): The `(num_params,N,N)`-shaped, matrix-by-vector
                derivative of this unitary at the point specified by params.

        Notes:
            The gradient of a unitary is defined as a matrix-by-vector
            derivative. If the UnitaryMatrix result of `get_unitary` has
            dimension NxN, then the shape of `get_grad`'s return value
            should equal (num_params,N,N), where the return value's
            i-th element is the matrix derivative of the unitary
            with respect to the i-th parameter.
        """

    def get_unitary_and_grad(
        self,
        params: Sequence[float] = [],
    ) -> tuple[UnitaryMatrix, np.ndarray]:
        """
        Return a tuple combining the outputs of `get_unitary` and `get_grad`.

        Args:
            params (Sequence[float]): The unitary parameters, see
                `Unitary.get_unitary` for more info.

        Returns:
            (UnitaryMatrix): The unitary matrix, see `Unitary.get_unitary`
                for more info.

            (np.ndarray): The unitary's gradient, see `get_grad`.

        Notes:
            Can be overridden to speed up optimization by calculating both
            at the same time.
        """
        return (self.get_unitary(params), self.get_grad(params))
