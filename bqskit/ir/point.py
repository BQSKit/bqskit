"""This module implements the CircuitPoint class."""
from __future__ import annotations

import logging
from typing import Any
from typing import NamedTuple
from typing import Tuple
from typing import Union

from typing_extensions import TypeGuard

from bqskit.utils.typing import is_integer
_logger = logging.getLogger(__name__)


class CircuitPoint(NamedTuple):
    """
    The CircuitPoint NamedTuple class.

    A CircuitPoint is a 2d-index into the Circuit data structure.
    """
    cycle: int
    qudit: int

    # TODO: Maybe add below code,
    # so we can do CircuitPoint(point) instead of CircuitPoint(*point)

    # def __new__(
    #     cls,
    #     cycle_or_tuple: int | tuple[int, int],
    #     qudit: int | None = None
    # ) -> CircuitPoint:
    #     if isinstance(cycle_or_tuple, tuple):
    #         if not CircuitPoint.is_point(cycle_or_tuple):
    #             raise TypeError("Expected two integer arguments.")

    #         qudit = cycle_or_tuple[1]
    #         cycle_or_tuple = cycle_or_tuple[0]

    #     if not is_integer(cycle_or_tuple) or not is_integer(qudit):
    #         raise TypeError("Expected two integer arguments.")

    #     super().__new__(cls, cycle_or_tuple, qudit)

    # def convert_to_positive(self, size: int, length: int) -> CircuitPoint:
    #     """Convert the point's indices to positive values."""
    #     return CircuitPoint(
    #         cycle if cycle > 0 else
    #     )

    @staticmethod
    def is_point(point: Any) -> TypeGuard[CircuitPointLike]:
        """Return true if point is a CircuitPointLike."""
        if isinstance(point, CircuitPoint):
            return True

        if not isinstance(point, tuple):
            _logger.debug('Point is not a tuple.')
            return False

        if len(point) != 2:
            _logger.debug(
                'Expected point to contain two values, got %d.' % len(point),
            )
            return False

        if not is_integer(point[0]):
            _logger.debug(
                'Expected integer values in point, got %s.' % type(point[0]),
            )
            return False

        if not is_integer(point[1]):
            _logger.debug(
                'Expected integer values in point, got %s.' % type(point[1]),
            )
            return False

        return True


CircuitPointLike = Union[Tuple[int, int], CircuitPoint]
