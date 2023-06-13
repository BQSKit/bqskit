"""This module implements the CostFunction base classes."""
from __future__ import annotations

import abc

from bqskit.qis.unitary.unitary import RealVector
from bqskit.utils.typing import is_real_number
from bqskit.utils.typing import is_sequence


class CostFunction(abc.ABC):
    """
    The CostFunction base class.

    A CostFunction is a map from a vector of real numbers to a real number. This
    output represents cost and minimizers will attempt to reduce it. The input
    should represent circuit parameters.
    """

    @abc.abstractmethod
    def get_cost(self, params: RealVector) -> float:
        """Return the cost value given the input parameters."""

    def __call__(self, params: RealVector) -> float:
        """Return the cost value given the input parameters."""

        if not is_sequence(params):
            raise TypeError(
                'Expected sequence for params, got %s.' % type(params),
            )

        if not all(is_real_number(param) for param in params):
            raise TypeError(
                'Expected sequence of floats for params.',
            )

        return self.get_cost(params)
