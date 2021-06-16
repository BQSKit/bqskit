"""This module implements the CircuitLocation class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Iterable
from typing import overload
from typing import Sequence
from typing import Union

from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_iterable
from bqskit.utils.typing import is_sequence
_logger = logging.getLogger(__name__)


class CircuitLocation(Sequence[int]):  # TODO: Consider making frozenset[int]
    """
    The CircuitLocation class.

    A CircuitLocation is an ordered set of qudit indices. In context, this
    usually describes where a gate or operation is being applied.
    """

    def __init__(self, location: int | Iterable[int]) -> None:
        """
        Construct a CircuitLocation from `location`.

        Args:
            location (int | Iterable[int]): The qudit indices.

        Raises:
            ValueError: If any qudit index is negative.

            ValueError: If there are duplicates in location.
        """

        if is_integer(location):  # TODO: Typeguard
            location = [location]  # type: ignore

        assert not isinstance(location, int)  # TODO: Typeguard
        location = list(location)

        if not is_iterable(location):
            raise TypeError(
                'Expected iterable of integers for location'
                f', got {type(location)}.',
            )

        if not all(is_integer(qudit_index) for qudit_index in location):
            checks = [is_integer(qudit_index) for qudit_index in location]
            raise TypeError(
                'Expected iterable of integers for location'
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
            return self._location[index]  # type: ignore  # TODO: TypeGuards

        if isinstance(index, slice):
            return self._location[index]  # type: ignore  # TODO: TypeGuards

        if not is_sequence(index):
            raise TypeError(f'Invalid index type, got {type(index)}.')

        # TODO: TypeGuards
        return tuple(self._location[idx] for idx in index)  # type: ignore

    def __len__(self) -> int:
        """Return the number of qudits described by the location."""
        return len(self._location)

    def union(self, other: CircuitLocationLike) -> CircuitLocation:
        """Return the location containing qudits from self or other."""
<<<<<<< HEAD
        return CircuitLocation(
            set(list(self._location) + list(other)),  # type: ignore
        )

    def intersection(self, other: CircuitLocationLike) -> CircuitLocation:
        """Return the location containing qudits from self and other."""
=======
        if is_integer(other):  # TODO: TypeGuard
            return CircuitLocation(self._location + [other])  # type: ignore  # noqa
        return CircuitLocation(set(self).union(CircuitLocation(other)))

    def intersection(self, other: CircuitLocationLike) -> CircuitLocation:
        """Return the location containing qudits from self and other."""
        if is_integer(other):  # TODO: TypeGuard
            return CircuitLocation(other if other in self else [])  # type: ignore  # noqa
>>>>>>> b7e98b8ed2337013990eb48247bdd3e5d5e3c9bd
        return CircuitLocation([x for x in self if x in other])  # type: ignore

    @staticmethod
    def is_location(location: Any, num_qudits: int | None = None) -> bool:
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
            (bool): True if the location is valid.
        """
        if isinstance(location, CircuitLocation):
            if num_qudits is not None:
                return max(location) < num_qudits
            return True

        if is_integer(location):
            location = [location]

        if not is_iterable(location):
            _logger.debug(
                'Expected iterable of integers for location'
                f', got {type(location)}.',
            )
            return False

        if not all(is_integer(qudit_index) for qudit_index in location):
            checks = [is_integer(qudit_index) for qudit_index in location]
            _logger.debug(
                'Expected iterable of integers for location'
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

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CircuitLocation):
            return self._location.__eq__(other._location)

        if is_integer(other):
            return len(self) == 1 and self[0] == other

        return self._location.__eq__(other)


CircuitLocationLike = Union[int, Iterable[int], CircuitLocation]
