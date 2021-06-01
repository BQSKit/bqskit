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
    def min_qudit(self) -> int:
        return min(self.keys())

    @property
    def max_qudit(self) -> int:
        return max(self.keys())

    @property
    def location(self) -> CircuitLocation:
        return CircuitLocation(self.keys())

    @property
    def points(self) -> list[CircuitPoint]:
        """Return the points described by this region."""
        return [
            CircuitPoint(cycle_index, qudit_index)
            for qudit_index, bounds in self.items()
            for cycle_index in bounds.indices
        ]

    def shift_left(self, amount_to_shift: int) -> None:
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

        self._bounds = {
            qudit_index:
            QuditBounds(bound[0] - amount_to_shift, bound[1] - amount_to_shift)
            for qudit_index, bound in self._bounds.items()
        }

    def shift_right(self, amount_to_shift: int) -> None:
        """
        Shift the region to the right by `amount_to_shift`.

        Args:
            amount_to_shift (int): Add `amount_to_shift` to all
                cycle indices.
        """
        self._bounds = {
            qudit_index:
            QuditBounds(bound[0] + amount_to_shift, bound[1] + amount_to_shift)
            for qudit_index, bound in self._bounds.items()
        }

    def overlaps(self, other: CircuitPointLike | CircuitRegionLike) -> bool:
        """Return true if `other` overlaps this region."""

        if CircuitPoint.is_point(other):  # TODO: Typeguard
            other = CircuitPoint(*other)  # type: ignore

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

        raise TypeError(
            'Expected either CircuitPoint or CircuitRegion, got %s.'
            % type(other),
        )

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
