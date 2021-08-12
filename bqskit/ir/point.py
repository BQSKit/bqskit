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
    The CircuitPoint NamedTuple class.

    A CircuitPoint is a 2d-index into the Circuit data structure.
    """

    def __new__(
        cls,
        cycle_or_tuple: int | tuple[int, int],
        qudit: int | None = None,
    ) -> CircuitPoint:
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

        # if cycle < 0:
        #     raise ValueError(
        #         'Expected cycle to be >= 0, got {cycle}.',
        #     )

        # if qudit < 0:
        #     raise ValueError(
        #         'Expected qudit to be >= 0, got {qudit}.',
        #     )

        return super().__new__(cls, (cycle, qudit))  # type: ignore

    @property
    def cycle(self) -> int:
        return self[0]

    @property
    def qudit(self) -> int:
        return self[1]

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
