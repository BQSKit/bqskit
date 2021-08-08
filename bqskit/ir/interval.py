"""This module implements the CycleInterval class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Iterable
from typing import Tuple
from typing import Union

from typing_extensions import TypeGuard

from bqskit.utils.typing import is_integer
_logger = logging.getLogger(__name__)


class CycleInterval(Tuple[int, int]):
    """
    The CycleInterval class.

    Contains an inclusive lower and upper cycle bound for a qudit.
    """

    def __new__(
        cls,
        lower_or_tuple: int | tuple[int, int],
        upper: int | None = None,
    ) -> CycleInterval:
        if upper is not None and not is_integer(upper):
            raise TypeError(
                f'Expected int or None for upper, got {type(upper)}.',
            )

        if isinstance(lower_or_tuple, tuple):
            if not CycleInterval.is_interval(lower_or_tuple):
                raise TypeError('Expected two integer arguments.')

            if upper is not None:
                raise ValueError('Unable to handle extra argument.')

            lower = lower_or_tuple[0]
            upper = lower_or_tuple[1]

        elif is_integer(lower_or_tuple) and is_integer(upper):
            lower = lower_or_tuple

        elif is_integer(lower_or_tuple) and upper is None:
            raise ValueError('Expected two integer arguments.')

        else:
            raise TypeError('Expected two integer arguments.')

        if lower > upper:
            raise ValueError(
                'Expected lower to be <= upper, got {lower} <= {upper}.',
            )

        if lower < 0:
            raise ValueError(
                'Expected positive integers, got {lower} and {upper}.',
            )

        return super().__new__(cls, (lower, upper))

    @property
    def lower(self) -> int:
        return self[0]

    @property
    def upper(self) -> int:
        return self[1]

    @property
    def indices(self) -> list[int]:
        return list(range(self.lower, self.upper + 1))

    def __contains__(self, cycle_index: object) -> bool:
        """Return true if `cycle_index` is inside this interval."""
        if not is_integer(cycle_index):
            return False

        return self.lower <= cycle_index <= self.upper

    def __iter__(self) -> Iterable[int]:
        return range(self.lower, self.upper + 1).__iter__()

    def __len__(self) -> int:
        return self.upper - self.lower + 1

    def overlaps(self, other: IntervalLike) -> bool:
        """Return true if `other` overlaps with this interval."""
        if not CycleInterval.is_interval(other):
            raise TypeError(f'Expected CycleInterval, got {type(other)}.')

        other = CycleInterval(other)

        return self.lower <= other.upper and self.upper >= other.lower

    def intersection(self, other: IntervalLike) -> CycleInterval:
        """Return the range defined by both `self` and `other` interval."""
        if not CycleInterval.is_interval(other):
            raise TypeError(f'Expected CycleInterval, got {type(other)}.')

        other = CycleInterval(other)

        if not self.overlaps(other):
            raise ValueError('Empty intersection in interval.')

        return CycleInterval(
            max(self.lower, other.lower),
            min(self.upper, other.upper),
        )

    def union(self, other: IntervalLike) -> CycleInterval:
        """Return the range defined by `self` or `other` interval."""
        if not CycleInterval.is_interval(other):
            raise TypeError(f'Expected CycleInterval, got {type(other)}.')

        other = CycleInterval(other)

        if not self.overlaps(other):
            raise ValueError('Union would lead to invalid interval.')

        return CycleInterval(
            min(self.lower, other.lower),
            max(self.upper, other.upper),
        )

    def __lt__(self, other: tuple[int, ...]) -> bool:
        """Defines Partial Ordering."""
        if CycleInterval.is_interval(other):
            return self.upper < other[0]

        # if isinstance(other, tuple):
        #     raise TypeError(f"Cannot compare {other} to CycleInterval.")

        return NotImplemented

    # def __le__(self, other: Tuple[int, ...]) -> bool:
    #     """Defines Partial Ordering"""
    #     if CycleInterval.is_interval(other):
    #         return self.upper < other[0] or self == other

    #     if isinstance(other, tuple):
    #         raise TypeError(f"Cannot compare {other} to CycleInterval.")

    #     return NotImplemented

    # def __gt__(self, other: Tuple[int, ...]) -> bool:
    #     """Defines Partial Ordering"""
    #     if CycleInterval.is_interval(other):
    #         return self.lower > other[1]

    #     if isinstance(other, tuple):
    #         raise TypeError(f"Cannot compare {other} to CycleInterval.")

    #     return NotImplemented

    # def __ge__(self, other: Tuple[int, ...]) -> bool:
    #     """Defines Partial Ordering"""
    #     if CycleInterval.is_interval(other):
    #         return self.lower > other[1] or self == other

    #     if isinstance(other, tuple):
    #         raise TypeError(f"Cannot compare {other} to CycleInterval.")

    #     return NotImplemented

    # def __eq__(self, other: Tuple[int, ...]) -> bool:
    #     """Defines Partial Ordering"""
    #     if CycleInterval.is_interval(other):
    #         return self.lower == other[0] and self.upper == other[1]

    #     if isinstance(other, tuple):
    #         raise TypeError(f"Cannot compare {other} to CycleInterval.")

    #     return NotImplemented

    # def __ne__(self, other: Tuple[int, ...]) -> bool:
    #     """Defines Partial Ordering"""
    #     if CycleInterval.is_interval(other):
    #         return self.lower != other[0] or self.upper != other[1]

    #     if isinstance(other, tuple):
    #         raise TypeError(f"Cannot compare {other} to CycleInterval.")

    #     return NotImplemented

    @staticmethod
    def is_interval(interval: Any) -> TypeGuard[IntervalLike]:
        """Return true if `interval` is a IntervalLike."""
        if isinstance(interval, CycleInterval):
            return True

        if not isinstance(interval, tuple):
            _logger.debug('Bounds is not a tuple.')
            return False

        if len(interval) != 2:
            _logger.debug(
                'Expected interval to contain two values, got %d.' % len(
                    interval,
                ),
            )
            return False

        if not is_integer(interval[0]):
            _logger.debug(
                'Expected integer values in interval, got %s.' % type(
                    interval[0],
                ),
            )
            return False

        if not is_integer(interval[1]):
            _logger.debug(
                'Expected integer values in interval, got %s.' % type(
                    interval[1],
                ),
            )
            return False

        return True

    def __repr__(self) -> str:
        return f'Interval(lower={self.lower}, upper={self.upper})'


IntervalLike = Union[Tuple[int, int], CycleInterval]
