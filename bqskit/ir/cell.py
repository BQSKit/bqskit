"""
This module implements the CircuitCell class.

A Circuit is composed of a grid of CircuitCells. A cell groups together
a gate and its parameters.
"""
from __future__ import annotations

from typing import Iterable
from typing import Sequence

from bqskit.ir.gate import Gate
from bqskit.ir.gates import FrozenParameterGate


class CircuitCell():
    """The CircuitCell class."""

    def __init__(
        self, gate: Gate,
        location: Iterable[int],
        params: Sequence[float] = [],
    ) -> None:
        """
        CircuitCell Constructor.
s
        Args:
            gate (Gate): The cell's gate.

            location (Iterable[int]):  The set of qudits this gate affects.

            params (Optional[Sequence[float]]): The parameters for the
                gate, if any.
        """
        self.gate = gate
        self.location = location
        self.params = list(params)

    def get_qasm(self) -> str:
        """Returns the qasm string for this operation."""

        if isinstance(self.gate, FrozenParameterGate):
            full_params = self.gate.get_full_params(self.params)
        else:
            full_params = self.params

        return '{}({}) q[{}];'.format(
            self.gate.get_qasm_name(),
            ', '.join([str(p) for p in full_params]),
            '], q['.join([str(q) for q in self.location]),
        )

    def __str__(self) -> str:
        pass  # TODO

    def __repr__(self) -> str:
        pass  # TODO
