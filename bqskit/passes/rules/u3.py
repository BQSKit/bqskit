"""This module implements the U3Decomposition."""
from __future__ import annotations

from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import U3Gate


class U3Decomposition(BasePass):
    """
    The U3Decomposition class.

    Convert a single-qubit circuit to U3 gate.
    """

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        if circuit.num_qudits != 1:
            raise ValueError(
                'Cannot convert multi-qudit circuit into ZXZXZ sequence.',
            )

        if circuit.radixes[0] != 2:
            raise ValueError(
                'Cannot convert non-qubit circuit into ZXZXZ sequence.',
            )

        utry = circuit.get_unitary()
        new_circuit = Circuit(1)
        new_circuit.append_gate(U3Gate(), 0, U3Gate.calc_params(utry))
        circuit.become(new_circuit)
