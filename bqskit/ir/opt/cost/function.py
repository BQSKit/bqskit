"""This module implements the CostFunction base classes."""

import abc
from bqskit.utils.typing import is_real_number, is_sequence
from typing import Sequence

class CostFunction(abc.ABC):
    """
    The CostFunction base class.

    A CostFunction is a map from a vector of real numbers to a real number.
    This output represents cost and minimizers will attempt to reduce it.
    The input should represent circuit parameters.
    """

    @abc.abstractmethod
    def get_cost(self, params: Sequence[float]) -> float:
        """Return the cost value given the input parameters."""
    
    def __call__(self, params: Sequence[float]) -> float:
        """Return the cost value given the input parameters."""

        if not is_sequence(params):
            raise TypeError(
                "Expected sequence for params, got %s." % type(params)
            )
        
        if not all(is_real_number(param) for param in params):
            raise TypeError(
                "Expected sequence of floats for params."
            )
        
        return self.get_cost(params)
