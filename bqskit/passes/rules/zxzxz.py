"""This module implements the ZXZXZDecomposition."""
from __future__ import annotations

import cmath

import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.sx import SqrtXGate
from bqskit.ir.gates.parameterized.rx import RXGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gates.parameterized.u1 import U1Gate


class ZXZXZDecomposition(BasePass):
    """
    The ZXZXZDecomposition class.

    Convert a single-qubit circuit to ZXZXZ sequence.
    """

    def __init__(
        self,
        always_use_rx: bool = False,
        always_use_u1: bool = False,
    ) -> None:
        """
        Construct a ZXZXZDecomposition pass.

        Args:
            always_use_rx (bool): If True, always use RX instead of SX.

            always_use_u1 (bool): If True, always use U1 instead of RZ.
        """

        if not isinstance(always_use_rx, bool):
            raise TypeError(
                f'Expected bool for always_use_rx, got {type(always_use_rx)}.',
            )

        if not isinstance(always_use_u1, bool):
            raise TypeError(
                f'Expected bool for always_use_u1, got {type(always_use_u1)}.',
            )

        self.always_use_rx = always_use_rx
        self.always_use_u1 = always_use_u1

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        if circuit.num_qudits != 1:
            raise ValueError(
                'Cannot convert multi-qudit circuit into ZXZXZ sequence.',
            )

        if circuit.radixes[0] != 2:
            raise ValueError(
                'Cannot convert non-qubit circuit into ZXZXZ sequence.',
            )

        # Decide on RX or SX
        no_sx = RXGate() in data.gate_set and SqrtXGate() not in data.gate_set
        use_rx = self.always_use_rx or no_sx

        # Decide on RZ or U1
        no_rz = U1Gate() in data.gate_set and RZGate() not in data.gate_set
        use_u1 = self.always_use_u1 or no_rz

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

        if use_u1:
            new_circuit.append_gate(U1Gate(), 0, [l])
        else:
            new_circuit.append_gate(RZGate(), 0, [l])

        if use_rx:
            new_circuit.append_gate(RXGate(), 0, [np.pi / 2])
        else:
            new_circuit.append_gate(SqrtXGate(), 0)

        if use_u1:
            new_circuit.append_gate(U1Gate(), 0, [t])
        else:
            new_circuit.append_gate(RZGate(), 0, [t])

        if use_rx:
            new_circuit.append_gate(RXGate(), 0, [np.pi / 2])
        else:
            new_circuit.append_gate(SqrtXGate(), 0)

        if use_u1:
            new_circuit.append_gate(U1Gate(), 0, [p])
        else:
            new_circuit.append_gate(RZGate(), 0, [p])

        circuit.become(new_circuit)
