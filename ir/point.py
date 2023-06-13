"""This module implements the CircuitPoint class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Tuple
from typing import Union

from typing_extensions import TypeGuard

from bqskit.utils.typing import is_integer
_logger = logging.getLogger(__name__)


class CircuitPoint(Tuple[int, int]):
    """
    A cycle and qudit index pair used to index a circuit.

    This is a subclass of a tuple, and therefore can be used where a tuple of
    two ints can be used.
    """

    def __new__(
        cls,
        cycle_or_tuple: int | tuple[int, int],
        qudit: int | None = None,
    ) -> CircuitPoint:
        """
        Construct a point.

        Args:
            cycle_or_tuple (int | tuple[int, int]): Either a cycle index
                given as an integer, or a tuple of a cycle index and qudit
                index. If an integer is given, then you will need to also
                specify the qudit index as the next argument.

            qudit (int | None): If `cycle_or_tuple` is an integer cycle
                index, then you will need to specify the qudit index here.
                Otherwise, leave this as None.

        Returns:
            CircuitPoint: The new point object.
        """

        if qudit is not None and not is_integer(qudit):
            raise TypeError(
                f'Expected int or None for qudit, got {type(qudit)}.',
            )

        if isinstance(cycle_or_tuple, tuple):
            if not CircuitPoint.is_point(cycle_or_tuple):
                raise TypeError('Expected two integer arguments.')

            if qudit is not None:
                raise ValueError('Unable to handle extra argument.')

            cycle = cycle_or_tuple[0]
            qudit = cycle_or_tuple[1]

        elif is_integer(cycle_or_tuple) and is_integer(qudit):
            cycle = cycle_or_tuple

        elif is_integer(cycle_or_tuple) and qudit is None:
            raise ValueError('Expected two integer arguments.')

        else:
            raise TypeError('Expected two integer arguments.')

        return super().__new__(cls, (cycle, qudit))  # type: ignore

    @property
    def cycle(self) -> int:
        """The point's cycle index."""
        return self[0]

    @property
    def qudit(self) -> int:
        """The point's qudit index."""
        return self[1]

    @staticmethod
    def is_point(point: Any) -> TypeGuard[CircuitPointLike]:
        """Return true if point is a CircuitPointLike."""
        if isinstance(point, CircuitPoint):
            return True

        if not isinstance(point, tuple):
            _logger.log(0, 'Point is not a tuple.')
            return False

        if len(point) != 2:
            _logger.log(
                0,
                'Expected point to contain two values, got %d.' % len(
                    point,
                ),
            )
            return False

        if not is_integer(point[0]):
            _logger.log(
                0,
                'Expected integer values in point, got %s.' % type(
                    point[0],
                ),
            )
            return False

        if not is_integer(point[1]):
            _logger.log(
                0,
                'Expected integer values in point, got %s.' % type(
                    point[1],
                ),
            )
            return False

        return True


CircuitPointLike = Union[Tuple[int, int], CircuitPoint]
