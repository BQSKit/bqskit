"""
This module implements the Circuit class.

A circuit represents a quantum program composed of gate objects.
"""
from __future__ import annotations

from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set

import numpy as np

from bqskit.ir.cell import CircuitCell
from bqskit.ir.gate import Gate
from bqskit.qis.unitary import Unitary


class Circuit(Unitary):
    """The Circuit class."""

    def __init__(
        self, num_qudits: int,
        qudit_radixes: Optional[List[int]] = None,
    ) -> None:
        """
        Circuit constructor. Builds an empty circuit with
        the specified number of qudits. By default, all qudits
        are qubits, but this can be changed with qudit_radixes parameter.

        Args:
            num_qudits (int): The number of qudits in this circuit.

            qudit_radixes (List[int]): A list with length equal
                to num_qudits. Each element specifies the base
                of a qudit. Defaults to qubits.

        Raises:
            ValueError: if num_qudits is non-positive.

        Examples:
            circ = Circuit(4) # Creates four-qubit empty circuit.
        """

        if not isinstance(num_qudits, int):
            raise TypeError(
                'Invalid type for num_qudits: '
                'expected int, got %s.' % type(num_qudits),
            )
        if num_qudits <= 0:
            raise ValueError('Expected positive number for num_qudits.')

        self.num_qudits = num_qudits
        self.qudit_radixes = qudit_radixes or [2 for q in range(num_qudits)]
        self._circuit: List[List[CircuitCell]] = [[]
                                                  for q in range(num_qudits)]
        self.gate_set: Set[Gate] = set()

    def get_num_params(self) -> int:
        pass

    def get_gate(self, qudit: int, time_step: int) -> Gate:
        pass

    def get_num_gates(self) -> int:
        pass

    def append_gate(self, gate: Gate, qudits: Iterable[int]) -> None:
        pass

    def remove_gate(self, qudit: int, time_step: int) -> None:
        pass

    def insert_gate(
        self, gate: Gate, qudits: Iterable[int],
        time_step: int,
    ) -> None:
        pass

    def get_unitary(self, params: Optional[List[float]] = None) -> np.ndarray:
        assert(params is None or len(params) == self.get_num_params())
        pass

    def __iter__(self) -> Iterator[Gate]:
        pass

    def __str__(self) -> str:
        pass

    def __add__(self, rhs: Circuit) -> Circuit:
        pass

    def __mul__(self, rhs: int) -> Circuit:
        pass

    def save(self, filename: str) -> None:
        pass

    @staticmethod
    def load(filename: str) -> Circuit:
        pass

    @staticmethod
    def from_str(str: str) -> Circuit:
        pass

    # staticmethod
    def from_unitary(utry: np.ndarray) -> Circuit:
        pass
