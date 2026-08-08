"""This module implements the PermutationGate."""
from __future__ import annotations

from collections.abc import Sequence

from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.qis.permutation import PermutationMatrix
from bqskit.utils.cachedclass import CachedClass


class PermutationGate(Gate, CachedClass):
    """A Permutation Gate."""

    _num_params = 0

    def __init__(
        self,
        num_qudits: int,
        location: Sequence[int],
    ) -> None:
        """
        Construct a gate that shifts the state of qudits around.

        See :func:PermutationMatrix.from_qubit_location for more.

        Args:
            num_qubits (int): Total number of qubits

            location (Sequence[int]): The desired locations to swap
                the starting qubits to.

        Raises:
            ValueError: If num_qudits is nonpositive.
        """
        if num_qudits <= 0:
            raise ValueError('Expected positive integer, got %d' % num_qudits)

        self._num_qudits = num_qudits
        self._radixes = tuple([2] * num_qudits)
        self.location = tuple(location)

        dim = 2 ** num_qudits
        entries = PermutationMatrix.from_qubit_location(
            num_qudits, self.location,
        ).numpy.real
        rows = [
            ','.join('1' if entries[r, c] > 0.5 else '0' for c in range(dim))
            for r in range(dim)
        ]
        self._expr = _UnitaryExpression(
            'Permutation<{}>() {{ [{}] }}'.format(
                ','.join(['2'] * num_qudits),
                ','.join('[%s]' % row for row in rows),
            ),
        )

    def __str__(self) -> str:
        return f'PermutationGate({self.location})'

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PermutationGate)
            and self.get_unitary() == other.get_unitary()
        )

    def __hash__(self) -> int:
        return hash(self.get_unitary())
