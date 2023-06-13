"""This module implements the PermutationGate."""
from __future__ import annotations

from typing import Sequence

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.permutation import PermutationMatrix


class PermutationGate(ConstantGate, QubitGate):
    """A Permutation Gate."""

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
        self.location = tuple(location)
        self._utry = PermutationMatrix.from_qubit_location(
            num_qudits, self.location,
        )

    def __str__(self) -> str:
        return f'PermutationGate({self.location})'

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PermutationGate)
            and self._utry == other._utry
        )

    def __hash__(self) -> int:
        return hash(self._utry)
