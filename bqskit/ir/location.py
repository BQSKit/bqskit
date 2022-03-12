"""This module implements the CircuitLocation class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Iterator
from typing import overload
from typing import Sequence
from typing import Union

from typing_extensions import TypeGuard

from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_iterable
from bqskit.utils.typing import is_mapping
from bqskit.utils.typing import is_sequence
from bqskit.utils.typing import is_sequence_of_int
_logger = logging.getLogger(__name__)


class CircuitLocation(Sequence[int]):
    """
    The CircuitLocation class.

    A CircuitLocation is an ordered set of qudit indices. In context, this
    usually describes where a gate or operation is being applied.
    """

    def __init__(self, location: int | Sequence[int]) -> None:
        """
        Construct a CircuitLocation from `location`.

        Args:
            location (int | Sequence[int]): The qudit indices.

        Raises:
            ValueError: If any qudit index is negative.

            ValueError: If there are duplicates in location.
        """

        if is_integer(location):
            location = [location]

        elif is_sequence(location):
            location = list(location)

        else:
            raise TypeError(
                f'Expected integer(s) for location, got {type(location)}.',
            )

        if not is_sequence_of_int(location):
            checks = [is_integer(q) for q in location]
            raise TypeError(
                'Expected iterable of positive integers for location'
                f', got atleast one {type(location[checks.index(False)])}.',
            )

        if not all(qudit_index >= 0 for qudit_index in location):
            checks = [qudit_index >= 0 for qudit_index in location]
            raise ValueError(
                'Expected iterable of positive integers for location'
                f', got atleast one {location[checks.index(False)]}.',
            )

        if len(set(location)) != len(location):
            raise ValueError('Location has duplicate qudit indices.')

        self._location: tuple[int, ...] = tuple(location)

    @overload
    def __getitem__(self, index: int) -> int:
        ...

    @overload
    def __getitem__(self, indices: slice | Sequence[int]) -> tuple[int, ...]:
        ...

    def __getitem__(
        self,
        index: int | slice | Sequence[int],
    ) -> int | tuple[int, ...]:
        """Retrieve one or multiple qudit indices from the location."""
        if is_integer(index):
            return self._location[index]

        if isinstance(index, slice):
            return self._location[index]

        if is_sequence_of_int(index):
            return tuple(self._location[idx] for idx in index)

        raise TypeError(f'Invalid index type, got {type(index)}.')

    def __len__(self) -> int:
        """Return the number of qudits described by the location."""
        return len(self._location)

    def union(self, other: CircuitLocationLike) -> CircuitLocation:
        """Return the location containing qudits from self or other."""
        if is_integer(other):
            if other in self:
                return self
            return CircuitLocation(self._location + (other,))
        return CircuitLocation(list(set(self).union(CircuitLocation(other))))

    def intersection(self, other: CircuitLocationLike) -> CircuitLocation:
        """Return the location containing qudits from self and other."""
        if is_integer(other):
            return CircuitLocation([other] if other in self else [])
        return CircuitLocation([x for x in self if x in other])  # type: ignore

    @staticmethod
    def is_location(
        location: Any,
        num_qudits: int | None = None,
    ) -> TypeGuard[CircuitLocationLike]:
        """
        Determines if the sequence of qudits form a valid location. A valid
        location is a set of qubit indices (integers) that are greater than or
        equal to zero, and if num_qudits is specified, less than num_qudits.

        Args:
            location (Any): The location to check.

            num_qudits (int | None): The total number of qudits.
                All qudit indices should be less than this. If None,
                don't check.

        Returns:
            (TypeGuard[CircuitLocationLike]): True if the location is valid.
        """
        if isinstance(location, CircuitLocation):
            if num_qudits is not None:
                return max(location) < num_qudits
            return True

        if is_mapping(location) or isinstance(location, set):
            return False

        if is_integer(location):
            location = [location]

        elif is_iterable(location):
            location = list(location)

        else:
            _logger.debug(
                f'Expected integer(s) for location, got {type(location)}.',
            )
            return False

        if not is_sequence_of_int(location):
            checks = [is_integer(q) for q in location]
            _logger.debug(
                'Expected iterable of positive integers for location'
                f', got atleast one {type(location[checks.index(False)])}.',
            )
            return False

        if not all(qudit_index >= 0 for qudit_index in location):
            checks = [qudit_index >= 0 for qudit_index in location]
            _logger.debug(
                'Expected iterable of positive integers for location'
                f', got atleast one {location[checks.index(False)]}.',
            )
            return False

        if len(set(location)) != len(location):
            _logger.debug('Location has duplicate qudit indices.')
            return False

        if num_qudits is not None:
            if not all([qudit < num_qudits for qudit in location]):
                _logger.debug('Location has an erroneously large qudit.')
                return False

        return True

    def __str__(self) -> str:
        return self._location.__str__()

    def __repr__(self) -> str:
        return self._location.__repr__()

    def __hash__(self) -> int:
        return self._location.__hash__()

    def __iter__(self) -> Iterator[int]:
        return self._location.__iter__()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CircuitLocation):
            return self._location.__eq__(other._location)

        if is_integer(other):
            return len(self) == 1 and self[0] == other

        return self._location.__eq__(other)


CircuitLocationLike = Union[int, Sequence[int], CircuitLocation]
