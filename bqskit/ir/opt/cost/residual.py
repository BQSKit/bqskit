"""This module implements the ResidualFunction base classes."""
from __future__ import annotations

import abc

import numpy as np
import numpy.typing as npt

from bqskit.ir.opt.cost.function import CostFunction
from bqskit.qis.unitary.unitary import RealVector
from bqskit.utils.typing import is_real_number
from bqskit.utils.typing import is_sequence


class ResidualsFunction(CostFunction):
    """
    The ResidualFunction base class.

    A ResidualFunction is a map from a vector of real numbers to a vector of
    real number. This output represents the residuals and minimizers will
    attempt to reduce it. The input should represent circuit parameters.
    """

    @abc.abstractmethod
    def get_residuals(self, params: RealVector) -> npt.NDArray[np.float64]:
        """Return the vector of residuals given the input parameters."""

    def __call__(  # type: ignore
        self,
        params: RealVector,
    ) -> npt.NDArray[np.float64]:
        """Return the vector of residuals given the input parameters."""

        if not is_sequence(params):
            raise TypeError(
                'Expected sequence for params, got %s.' % type(params),
            )

        if not all(is_real_number(param) for param in params):
            raise TypeError(
                'Expected sequence of floats for params.',
            )

        return self.get_residuals(params)
