"""This module implements the CycleInterval class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Iterator
from typing import Tuple
from typing import Union

from typing_extensions import TypeGuard

from bqskit.utils.typing import is_integer
_logger = logging.getLogger(__name__)


class CycleInterval(Tuple[int, int]):
    """
    The CycleInterval class.

    Represents an inclusive range of cycles in a given circuit.
    """

    def __new__(
        cls,
        lower_or_tuple: int | tuple[int, int],
        upper: int | None = None,
    ) -> CycleInterval:
        """
        CycleInterval Constructor.

        Allows constructing a CycleInterval with either a tuple of ints
        or two ints.

        Args:
            lower_or_tuple (int | tuple[int, int]): Either the lower
                bound for the interval or the tuple of lower and upper
                bounds.

            upper (int | None): The upper bound for the interval. If a
                tuple is passed in for `lower_or_tuple` then this should
                be None.

        Returns:
            (CycleInterval): The constructed CycleInterval.

        Raises:
            ValueError: If `lower_or_tuple` is a tuple and 'upper' is
                not None, or if `lower_or_tuple` is an integer and
                `upper` is None.

            ValueError: If the lower bound is greater than the upper bound.

            ValueError: If either bound is negative.

        Notes:
            The lower and upper bounds are inclusive.
        """
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

        return super().__new__(cls, (lower, upper))  # type: ignore

    @property
    def lower(self) -> int:
        """The interval's inclusive lower bound."""
        return self[0]

    @property
    def upper(self) -> int:
        """The interval's inclusive upper bound."""
        return self[1]

    @property
    def indices(self) -> list[int]:
        """The indices contained within the interval."""
        return list(range(self.lower, self.upper + 1))

    def __contains__(self, cycle_index: object) -> bool:
        """Return true if `cycle_index` is inside this interval."""
        return self.lower <= cycle_index <= self.upper  # type: ignore

    def __iter__(self) -> Iterator[int]:
        """Return an iterator for all indices contained in the interval."""
        return range(self.lower, self.upper + 1).__iter__()

    def __len__(self) -> int:
        """The length of the interval."""
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

        if not self.overlaps(other) and (
            self.upper + 1 != other[0]
            and self.lower - 1 != other[1]
        ):
            raise ValueError('Union would lead to invalid interval.')

        return CycleInterval(
            min(self.lower, other.lower),
            max(self.upper, other.upper),
        )

    def __lt__(self, other: tuple[int, ...]) -> bool:
        """
        Return true if `self` comes before `other`.

        The less than operator defines a partial ordering.
        """
        if CycleInterval.is_interval(other):
            return self.upper < other[0]

        return NotImplemented

    @staticmethod
    def is_interval(interval: Any) -> TypeGuard[IntervalLike]:
        """Return true if `interval` is a IntervalLike."""
        if isinstance(interval, CycleInterval):
            return True

        if not isinstance(interval, tuple):
            _logger.log(0, 'Bounds is not a tuple.')
            return False

        if len(interval) != 2:
            _logger.log(
                0,
                'Expected interval to contain two values'
                f', got {len(interval)}.',
            )
            return False

        if not is_integer(interval[0]):
            _logger.log(
                0,
                'Expected integer values in interval'
                f', got {type(interval[0])}.',
            )
            return False

        if not is_integer(interval[1]):
            _logger.log(
                0,
                'Expected integer values in interval'
                f', got {type(interval[1])}.',
            )
            return False

        return True

    def __repr__(self) -> str:
        """Return a string representation of the interval."""
        return f'Interval(lower={self.lower}, upper={self.upper})'


IntervalLike = Union[Tuple[int, int], CycleInterval]
