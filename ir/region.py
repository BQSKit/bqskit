"""This module implements the CircuitRegion class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Iterator
from typing import Mapping
from typing import Union

from typing_extensions import TypeGuard

from bqskit.ir.interval import CycleInterval
from bqskit.ir.interval import IntervalLike
from bqskit.ir.location import CircuitLocation
from bqskit.ir.point import CircuitPoint
from bqskit.ir.point import CircuitPointLike
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_mapping
_logger = logging.getLogger(__name__)


class CircuitRegion(Mapping[int, CycleInterval]):
    """
    The CircuitRegion class.

    A CircuitRegion is an contiguous, convex area in a circuit. It is
    represented as a map from qudit indices to cycle intervals.
    """

    def __init__(self, intervals: Mapping[int, IntervalLike]) -> None:
        """
        CircuitRegion Initializer.

        Args:
            intervals (Mapping[int, IntervalLike]): A map from qudit
                indices to cycle intervals. The cycle intervals can
                be given as either a CycleInterval object or a tuple
                of two ints that represent the lower and upper bound
                of the inclusive interval.

        Notes:
            All cycle intervals are inclusive.
        """
        if not is_mapping(intervals):
            raise TypeError(
                f'Expected mapping from int to IntervalLike, got {intervals}.',
            )

        for qudit, interval in intervals.items():
            if not is_integer(qudit):
                raise TypeError(f'Expected integer keys, got {qudit}.')

            if not CycleInterval.is_interval(interval):
                raise TypeError(
                    f'Expected valid CycleInterval, got {interval}.',
                )

        self._intervals = {
            qudit: CycleInterval(interval)
            for qudit, interval in intervals.items()
        }

    def __getitem__(self, key: int) -> CycleInterval:
        """
        Return the interval given by the qudit `key`.

        Raises:
            IndexError: If `key` is not in this region.
        """
        return self._intervals[key]

    def __iter__(self) -> Iterator[int]:
        """Return an iterator for all qudits contained in the region."""
        return iter(self._intervals)

    def __len__(self) -> int:
        """Return the number of qudits in the region."""
        return len(self._intervals)

    @property
    def min_cycle(self) -> int:
        """
        Return the smallest cycle index for any qudit.

        Raises:
            ValueError: If `self` is empty.
        """
        if self.empty:
            raise ValueError('Empty region cannot have minimum cycle.')

        return min(interval.lower for interval in self.values())

    @property
    def max_cycle(self) -> int:
        """
        Return the largest cycle index for any qudit.

        Raises:
            ValueError: If `self` is empty.
        """
        if self.empty:
            raise ValueError('Empty region cannot have maximum cycle.')

        return max(interval.upper for interval in self.values())

    @property
    def max_min_cycle(self) -> int:
        """
        Return the largest cycle index for any lower bound.

        Raises:
            ValueError: If `self` is empty.
        """
        if self.empty:
            raise ValueError('Empty region cannot have minimum cycle.')

        return max(interval.lower for interval in self.values())

    @property
    def min_max_cycle(self) -> int:
        """
        Return the smallest cycle index for any upper bound.

        Raises:
            ValueError: If `self` is empty.
        """
        if self.empty:
            raise ValueError('Empty region cannot have maximum cycle.')

        return min(interval.upper for interval in self.values())

    @property
    def min_qudit(self) -> int:
        """
        Return the smallest qudit index.

        Raises:
            ValueError: If `self` is empty.
        """
        if self.empty:
            raise ValueError('Empty region cannot have minimum qudit.')

        return min(self.keys())

    @property
    def max_qudit(self) -> int:
        """
        Return the largest qudit index.

        Raises:
            ValueError: If `self` is empty.
        """
        if self.empty:
            raise ValueError('Empty region cannot have maximum qudit.')

        return max(self.keys())

    @property
    def location(self) -> CircuitLocation:
        """Return the qudits this region includes as a CircuitLocation."""
        return CircuitLocation(sorted(self.keys()))

    @property
    def points(self) -> list[CircuitPoint]:
        """Return the points described by this region."""
        return [
            CircuitPoint(cycle_index, qudit_index)
            for qudit_index, intervals in self.items()
            for cycle_index in intervals.indices
        ]

    @property
    def volume(self) -> int:
        """Return the volume of this region measured in qudit-cycles."""
        return sum(len(interval) for interval in self.values())

    @property
    def width(self) -> int:
        """Return the number of cycles this region participates in."""
        if self.empty:
            return 0

        return self.max_cycle - self.min_cycle + 1

    @property
    def empty(self) -> bool:
        """Return true if this region is empty."""
        return len(self) == 0

    def shift_left(self, amount_to_shift: int) -> CircuitRegion:
        """
        Shift the region to the left by `amount_to_shift`.

        Args:
            amount_to_shift (int): Subtract `amount_to_shift` from all
                cycle indices.

        Raises:
            ValueError: If the region would be shifted below zero as a
                result of performing this operation.
        """
        if not is_integer(amount_to_shift):
            raise TypeError(
                f'Expected integer for shift amount, got {amount_to_shift}.',
            )

        if self.empty:
            return CircuitRegion(self._intervals)

        if self.min_cycle - amount_to_shift < 0:
            raise ValueError(
                f'Cannot shift region to the left by {amount_to_shift}.',
            )

        return CircuitRegion({
            qudit_index:
            CycleInterval(
                interval[0] - amount_to_shift,
                interval[1] - amount_to_shift,
            )
            for qudit_index, interval in self._intervals.items()
        })

    def shift_right(self, amount_to_shift: int) -> CircuitRegion:
        """
        Shift the region to the right by `amount_to_shift`.

        Args:
            amount_to_shift (int): Add `amount_to_shift` to all
                cycle indices.
        """
        if not is_integer(amount_to_shift):
            raise TypeError(
                f'Expected integer for shift amount, got {amount_to_shift}.',
            )

        if self.empty:
            return CircuitRegion(self._intervals)

        if amount_to_shift < 0:
            return self.shift_left(-amount_to_shift)

        return CircuitRegion({
            qudit_index:
            CycleInterval(
                interval[0] + amount_to_shift,
                interval[1] + amount_to_shift,
            )
            for qudit_index, interval in self._intervals.items()
        })

    def overlaps(self, other: CircuitPointLike | CircuitRegionLike) -> bool:
        """Return true if `other` overlaps this region."""

        if CircuitPoint.is_point(other):
            other = CircuitPoint(*other)

            if other.qudit not in self:
                return False

            intervals = self[other.qudit]
            return intervals.lower <= other.cycle <= intervals.upper

        if CircuitRegion.is_region(other):
            other = CircuitRegion(other)

            if other.empty or self.empty:
                return False

            if self.min_cycle > other.max_cycle:
                return False

            if self.max_cycle < other.min_cycle:
                return False

            qudit_intersection = self.location.intersection(other.location)

            if len(qudit_intersection) == 0:
                return False

            for qudit in qudit_intersection:
                if (
                    self[qudit].lower <= other[qudit].upper
                    and self[qudit].upper >= other[qudit].lower
                ):
                    return True

            return False

        raise TypeError(
            'Expected either CircuitPoint or CircuitRegion, got %s.'
            % type(other),
        )

    def __contains__(self, other: object) -> bool:
        if is_integer(other):
            return other in self._intervals.keys()

        if CircuitPoint.is_point(other):
            return other[1] in self.keys() and other[0] in self[other[1]]

        if CircuitRegion.is_region(other):
            other = CircuitRegion(other)

            if other.empty:
                return True

            if self.empty:
                return False

            return (
                all(qudit in self.keys() for qudit in other.keys())
                and all(
                    self[qudit].lower <= other[qudit][0] <= self[qudit].upper
                    for qudit in other.keys()
                )
                and all(
                    self[qudit].lower <= other[qudit][1] <= self[qudit].upper
                    for qudit in other.keys()
                )
            )

        return NotImplemented

    def transpose(self) -> dict[int, list[int]]:
        """Flip region to map cycle indices to qudit indices."""
        if self.empty:
            return {}

        qudit_cycles: dict[int, list[int]] = {
            i: []
            for i in range(self.min_cycle, self.max_cycle + 1)
        }

        for qudit_index, intervals in sorted(self.items()):
            for cycle_index in range(intervals.lower, intervals.upper + 1):
                qudit_cycles[cycle_index].append(qudit_index)

        return {k: v for k, v in qudit_cycles.items() if len(v) > 0}

    def intersection(self, other: CircuitRegionLike) -> CircuitRegion:
        if not CircuitRegion.is_region(other):
            raise TypeError(f'Expected CircuitRegion, got {type(other)}.')

        other = CircuitRegion(other)
        location = self.location.intersection(other.location)

        region: dict[int, CycleInterval] = {}
        for qudit in location:
            if self[qudit].overlaps(other[qudit]):
                region[qudit] = self[qudit].intersection(other[qudit])

        return CircuitRegion(region)

    def union(self, other: CircuitRegionLike) -> CircuitRegion:
        if not CircuitRegion.is_region(other):
            raise TypeError(f'Expected CircuitRegion, got {type(other)}.')

        other = CircuitRegion(other)
        location = self.location.union(other.location)

        region: dict[int, CycleInterval] = {}
        for qudit in location:
            if qudit in self and qudit in other:
                region[qudit] = self[qudit].union(other[qudit])

            elif qudit in self:
                region[qudit] = self[qudit]

            elif qudit in other:
                region[qudit] = other[qudit]

        return CircuitRegion(region)

    def depends_on(self, other: CircuitRegionLike) -> bool:
        """Return true if self depends on other."""
        if not isinstance(other, CircuitRegion):
            other = CircuitRegion(other)

        intersection = self.location.intersection(other.location)
        if len(intersection) != 0:
            return all(other[qudit] < self[qudit] for qudit in intersection)

        return False

    def dependency(self, other: CircuitRegionLike) -> int:
        """
        Return 1 if self depends on other.

        Return -1 if other depends on self. Return 0 if no dependency.
        """

        if not isinstance(other, CircuitRegion):
            other = CircuitRegion(other)

        intersection = self.location.intersection(other.location)

        if not intersection:
            return 0

        for qudit in intersection:
            if other[qudit] < self[qudit]:
                return 1

        return -1

    def __eq__(self, other: object) -> bool:
        if CircuitRegion.is_region(other):
            other = CircuitRegion(other)
            return sorted(self.items()) == sorted(other.items())

        return NotImplemented

    def __hash__(self) -> int:
        return tuple(sorted(self.items())).__hash__()

    def __str__(self) -> str:
        return str(self._intervals)

    def __repr__(self) -> str:
        return repr(self._intervals)

    def __lt__(self, other: object) -> bool:
        if CircuitPoint.is_point(other):
            other = CircuitPoint(*other)
            if other[0] < self.min_cycle:
                return True

            if other[1] in self.keys():
                return other[0] < self[other[1]].lower

        elif CircuitRegion.is_region(other):
            other = CircuitRegion(other)

            if len(self.location.intersection(other.location)) != 0:
                lt = None
                for qudit in self.location.intersection(other.location):
                    if lt is None:
                        lt = self[qudit] < other[qudit]
                    elif lt != (self[qudit] < other[qudit]):
                        raise ValueError('Both regions depend on each other.')

                assert lt is not None
                return lt

            lower_intervals = tuple(sorted({x.lower for x in self.values()}))
            other_lower_intervals = tuple(
                sorted({x.lower for x in other.values()}),
            )
            upper_intervals = tuple(
                reversed(sorted({x.upper for x in self.values()})),
            )
            other_upper_intervals = tuple(
                reversed(sorted({x.upper for x in other.values()})),
            )
            return (lower_intervals, upper_intervals) < (
                other_lower_intervals, other_upper_intervals,
            )

        return NotImplemented

    @staticmethod
    def is_region(region: Any) -> TypeGuard[CircuitRegionLike]:
        """Return true if region is a CircuitRegionLike."""
        if isinstance(region, CircuitRegion):
            return True

        if not is_mapping(region):
            _logger.log(0, 'Region is not a mapping.')
            return False

        if not all(is_integer(key) for key in region.keys()):
            _logger.log(0, 'Region does not have integer keys.')
            return False

        if not all(CycleInterval.is_interval(val) for val in region.values()):
            _logger.log(0, 'Region does not have IntervalLike values.')
            return False

        return True


CircuitRegionLike = Union[Mapping[int, IntervalLike], CircuitRegion]
