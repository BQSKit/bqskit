"""This module implements the CircuitRegion class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Iterator
from typing import Mapping
from typing import NamedTuple
from typing import Tuple
from typing import Union

from bqskit.ir.location import CircuitLocation
from bqskit.ir.point import CircuitPoint
from bqskit.ir.point import CircuitPointLike
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_mapping
_logger = logging.getLogger(__name__)


# TODO: maybe change to CycleRange(Tuple[int, int])
class QuditBounds(NamedTuple):
    """
    The QuditBounds NamedTuple.

    Contains an inclusive lower and upper cycle bound for a qudit.
    """
    lower: int
    upper: int

    @property
    def indices(self) -> list[int]:
        return list(range(self.lower, self.upper + 1))

    def __contains__(self, cycle_index: object) -> bool:
        """Return true if `cycle_index` is inside this bounds."""
        if not is_integer(cycle_index):
            return False

        assert isinstance(cycle_index, int)  # TODO: Typeguards

        return self.lower <= cycle_index <= self.upper

    def overlaps(self, other: QuditBoundsLike) -> bool:
        """Return true if `other` overlaps with this bounds."""
        if not QuditBounds.is_bounds(other):
            raise TypeError(f'Expected QuditBounds, got {type(other)}.')

        other = QuditBounds(*other)

        return self.lower < other.upper and self.upper > other.lower

    def intersection(self, other: QuditBoundsLike) -> QuditBounds:
        """Return the range defined by both `self` and `other` bounds."""
        if not QuditBounds.is_bounds(other):
            raise TypeError(f'Expected QuditBounds, got {type(other)}.')

        other = QuditBounds(*other)

        if not self.overlaps(other):
            raise ValueError('Empty intersection in bounds.')

        return QuditBounds(
            max(self.lower, other.lower),
            min(self.upper, other.upper),
        )

    def union(self, other: QuditBoundsLike) -> QuditBounds:
        """Return the range defined by `self` or `other` bounds."""
        if not QuditBounds.is_bounds(other):
            raise TypeError(f'Expected QuditBounds, got {type(other)}.')

        other = QuditBounds(*other)

        if not self.overlaps(other):
            raise ValueError('Union would lead to invalid bounds.')

        return QuditBounds(
            min(self.lower, other.lower),
            max(self.upper, other.upper),
        )

    def __lt__(self, other: object) -> bool:
        if QuditBounds.is_bounds(other):  # TODO: TypeGuard
            return self.upper < other[0]  # type: ignore
        return NotImplemented

    @staticmethod
    def is_bounds(bounds: Any) -> bool:
        """Return true if bounds is a QuditBoundsLike."""
        if isinstance(bounds, QuditBounds):
            return True

        if not isinstance(bounds, tuple):
            _logger.debug('Bounds is not a tuple.')
            return False

        if len(bounds) != 2:
            _logger.debug(
                'Expected bounds to contain two values, got %d.' % len(bounds),
            )
            return False

        if not is_integer(bounds[0]):
            _logger.debug(
                'Expected integer values in bounds, got %s.' % type(bounds[0]),
            )
            return False

        if not is_integer(bounds[1]):
            _logger.debug(
                'Expected integer values in bounds, got %s.' % type(bounds[1]),
            )
            return False

        if bounds[1] < bounds[0]:
            _logger.debug('Upper bound is less than lower bound.')
            return False

        return True


class CircuitRegion(Mapping[int, QuditBounds]):
    """
    The CircuitRegion class.

    A CircuitRegion is an contiguous area in a circuit. It is represented as a
    map from qudit indices to lower and upper bounds given by cycle indices.
    """

    def __init__(self, bounds: Mapping[int, QuditBoundsLike]) -> None:
        self._bounds = {
            qudit: QuditBounds(*bound)
            for qudit, bound in bounds.items()
        }

    def __getitem__(self, key: int) -> QuditBounds:
        return self._bounds[key]

    def __iter__(self) -> Iterator[int]:
        return iter(self._bounds)

    def __len__(self) -> int:
        return len(self._bounds)

    @property
    def min_cycle(self) -> int:
        return min(bound[0] for bound in self.values())

    @property
    def max_cycle(self) -> int:
        return max(bound[1] for bound in self.values())

    @property
    def max_min_cycle(self) -> int:
        return max(bound[0] for bound in self.values())

    @property
    def min_max_cycle(self) -> int:
        return min(bound[1] for bound in self.values())

    @property
    def min_qudit(self) -> int:
        return min(self.keys())

    @property
    def max_qudit(self) -> int:
        return max(self.keys())

    @property
    def location(self) -> CircuitLocation:
        return CircuitLocation(sorted(self.keys()))

    @property
    def points(self) -> list[CircuitPoint]:
        """Return the points described by this region."""
        return [
            CircuitPoint(cycle_index, qudit_index)
            for qudit_index, bounds in self.items()
            for cycle_index in bounds.indices
        ]

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
        if self.min_cycle - amount_to_shift < 0:
            raise ValueError(
                f'Cannot shift region to the left by {amount_to_shift}.',
            )

        return CircuitRegion({
            qudit_index:
            QuditBounds(bound[0] - amount_to_shift, bound[1] - amount_to_shift)
            for qudit_index, bound in self._bounds.items()
        })

    def shift_right(self, amount_to_shift: int) -> CircuitRegion:
        """
        Shift the region to the right by `amount_to_shift`.

        Args:
            amount_to_shift (int): Add `amount_to_shift` to all
                cycle indices.
        """
        return CircuitRegion({
            qudit_index:
            QuditBounds(bound[0] + amount_to_shift, bound[1] + amount_to_shift)
            for qudit_index, bound in self._bounds.items()
        })

    def overlaps(self, other: CircuitPointLike | CircuitRegionLike) -> bool:
        """Return true if `other` overlaps this region."""

        if CircuitPoint.is_point(other):
            other = CircuitPoint(*other)

            if other.qudit not in self:
                return False

            bounds = self[other.qudit]
            return bounds.lower <= other.cycle <= bounds.upper

        if CircuitRegion.is_region(other):  # TODO: Typeguard
            other = CircuitRegion(other)  # type: ignore

            if self.min_cycle > other.max_cycle:
                return False

            if self.max_cycle < other.min_cycle:
                return False

            qudit_intersection = self.location.intersection(other.location)

            if len(qudit_intersection) == 0:
                return False

            for qudit in qudit_intersection:
                if (
                    self[qudit].lower < other[qudit].upper
                    and self[qudit].upper > other[qudit].lower
                ):
                    return True

            return False

        raise TypeError(
            'Expected either CircuitPoint or CircuitRegion, got %s.'
            % type(other),
        )

    def __contains__(self, other: object) -> bool:
        if is_integer(other):
            return other in self._bounds.keys()

        if CircuitPoint.is_point(other):  # TODO: TypeGuard
            return other[1] in self.keys() and other[0] in self[other[1]]  # type: ignore  # noqa

        if CircuitRegion.is_region(other):  # TODO: TypeGuard
            other = CircuitRegion(other)  # type: ignore
            return (
                all(qudit in self.keys() for qudit in other.keys())
                and all(
                    self[qudit].lower <= other[qudit][0] <= self[qudit].upper
                    for qudit in self.keys()
                )
                and all(
                    self[qudit].lower <= other[qudit][1] <= self[qudit].upper
                    for qudit in self.keys()
                )
            )

        return NotImplemented

    def transpose(self) -> dict[int, list[int]]:
        """Flip region to map cycle indices to qudit indices."""
        qudit_cycles: dict[int, list[int]] = {
            i: []
            for i in range(self.min_cycle, self.max_cycle + 1)
        }

        for qudit_index, bounds in sorted(self.items()):
            for cycle_index in range(bounds.lower, bounds.upper + 1):
                qudit_cycles[cycle_index].append(qudit_index)

        return {k: v for k, v in qudit_cycles.items() if len(v) > 0}

    def intersection(self, other: CircuitRegionLike) -> CircuitRegion:
        if not CircuitRegion.is_region(other):
            raise TypeError(f'Expected CircuitRegion, got {type(other)}.')

        other = CircuitRegion(other)
        location = self.location.intersection(other.location)

        region: dict[int, QuditBounds] = {}
        for qudit in location:
            if self[qudit].overlaps(other[qudit]):
                region[qudit] = self[qudit].intersection(other[qudit])

        return CircuitRegion(region)

    def union(self, other: CircuitRegionLike) -> CircuitRegion:
        if not CircuitRegion.is_region(other):
            raise TypeError(f'Expected CircuitRegion, got {type(other)}.')

        other = CircuitRegion(other)
        location = self.location.union(other.location)

        region: dict[int, QuditBounds] = {}
        for qudit in location:
            if qudit in self and qudit in other:
                region[qudit] = self[qudit].union(other[qudit])

            elif qudit in self:
                region[qudit] = self[qudit]

            elif qudit in other:
                region[qudit] = other[qudit]

        return CircuitRegion(region)

    def __eq__(self, other: object) -> bool:
        if CircuitRegion.is_region(other):  # TODO: TypeGuard
            other = CircuitRegion(other)  # type: ignore
            return sorted(self.items()) == sorted(other.items())

        return NotImplemented

    def __hash__(self) -> int:
        return tuple(sorted(self.items())).__hash__()

    def __str__(self) -> str:
        return str(self._bounds)

    def __repr__(self) -> str:
        return repr(self._bounds)

    def __lt__(self, other: object) -> bool:
        if CircuitPoint.is_point(other):  # TODO: TypeGuard
            other = CircuitPoint(*other)  # type: ignore
            if other[0] < self.min_cycle:
                return True

            if other[1] in self.keys():
                return other[0] < self[other[1]].lower

        elif CircuitRegion.is_region(other):  # TODO: TypeGuard
            other = CircuitRegion(other)  # type: ignore

            if len(self.location.intersection(other.location)) != 0:
                lt = None
                for qudit in self.location.intersection(other.location):
                    if lt is None:
                        lt = self[qudit] < other[qudit]
                    elif lt != self[qudit] < other[qudit]:
                        raise ValueError('Both regions depend on each other.')

                assert lt is not None
                return lt

            lower_bounds = tuple(sorted({x.lower for x in self.values()}))
            other_lower_bounds = tuple(
                sorted({x.lower for x in other.values()}),
            )
            upper_bounds = tuple(
                reversed(sorted({x.upper for x in self.values()})),
            )
            other_upper_bounds = tuple(
                reversed(sorted({x.upper for x in other.values()})),
            )
            return (lower_bounds, upper_bounds) < (
                other_lower_bounds, other_upper_bounds,
            )

        return NotImplemented

    @staticmethod
    def is_region(region: Any) -> bool:
        """Return true if region is a CircuitRegionLike."""
        if isinstance(region, CircuitRegion):
            return True

        if not is_mapping(region):
            _logger.debug('Region is not a mapping.')
            return False

        if not all(is_integer(key) for key in region.keys()):
            _logger.debug('Region does not have integer keys.')
            return False

        if not all(QuditBounds.is_bounds(val) for val in region.values()):
            _logger.debug('Region does not have QuditBoundsLike values.')
            return False

        return True


QuditBoundsLike = Union[Tuple[int, int], QuditBounds]
CircuitRegionLike = Union[Mapping[int, QuditBoundsLike], CircuitRegion]
