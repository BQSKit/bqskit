"""This module implements the CircuitLocation class."""
from __future__ import annotations

from typing import overload
from typing import Sequence
from typing import Union

from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_sequence


class CircuitLocation(Sequence[int]):
    """
    The CircuitLocation class.

    A CircuitLocation is an ordered set of qudit indices. In context, this
    usually describes where a gate or operation is being applied.
    """

    def __init__(self, location: Sequence[int]) -> None:
        """Construct a CircuitLocation from `location`."""

        if not is_sequence(location):
            raise TypeError(
                'Expected sequence of integers for location'
                f', got {type(location)}.',
            )

        if not all(is_integer(qudit_index) for qudit_index in location):
            truth_vals = [is_integer(qudit_index) for qudit_index in location]
            raise TypeError(
                'Expected sequence of integers for location'
                f', got atleast one {type(location[truth_vals.index(False)])}',
            )

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


CircuitLocationLike = Union[Sequence[int], CircuitLocation]
