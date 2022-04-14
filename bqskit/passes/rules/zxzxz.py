"""This module implements the ZXZXZDecomposition."""
from __future__ import annotations

import cmath
from typing import Any

import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import SqrtXGate


class ZXZXZDecomposition(BasePass):
    """
    The ZXZXZDecomposition class.

    Convert a single-qubit circuit to ZXZXZ sequence.
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

        # Calculate params
        utry = np.linalg.det(utry) ** (-0.5) * utry
        i1 = cmath.phase(utry[1, 1])
        i2 = cmath.phase(utry[1, 0])
        t = 2 * np.arctan2(abs(utry[1, 0]), abs(utry[0, 0])) + np.pi
        p = i1 + i2 + np.pi
        l = i1 - i2

        # Move angles into [-pi, pi)
        t = (t + np.pi) % (2 * np.pi) - np.pi
        p = (p + np.pi) % (2 * np.pi) - np.pi
        l = (l + np.pi) % (2 * np.pi) - np.pi

        new_circuit = Circuit(1)
        new_circuit.append_gate(RZGate(), 0, [l])
        new_circuit.append_gate(SqrtXGate(), 0)
        new_circuit.append_gate(RZGate(), 0, [t])
        new_circuit.append_gate(SqrtXGate(), 0)
        new_circuit.append_gate(RZGate(), 0, [p])
        circuit.become(new_circuit)
