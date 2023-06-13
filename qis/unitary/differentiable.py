"""This module implements the DifferentiableUnitary base class."""
from __future__ import annotations

import abc

import numpy as np
import numpy.typing as npt

from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitary import Unitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class DifferentiableUnitary(Unitary):
    """
    A differentiable unitary-valued function.

    A `DifferentiableUnitary` inherits from `Unitary` and additionally
    exposes the :func:`get_grad` abstract method and the
    :func:`get_unitary_and_grad` method.
    """

    @abc.abstractmethod
    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for the unitary map as an np.ndarray.

        Args:
            params (RealVector): The unitary parameters, see
                :func:`Unitary.get_unitary` for more info.

        Returns:
            np.ndarray: The `(num_params,N,N)`-shaped, matrix-by-vector
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
        params: RealVector = [],
    ) -> tuple[UnitaryMatrix, npt.NDArray[np.complex128]]:
        """
        Return a tuple combining the outputs of `get_unitary` and `get_grad`.

        Args:
            params (RealVector): The unitary parameters, see
                :func:`Unitary.get_unitary` for more info.

        Returns:
            tuple: tuple containing:
                UnitaryMatrix: The unitary matrix, see
                :func:`Unitary.get_unitary` for more info.

                np.ndarray: The unitary's gradient, see :func:`get_grad`.

        Notes:
            Can be overridden to potentially speed up optimization by
            calculating both at the same time.
        """
        return (self.get_unitary(params), self.get_grad(params))
