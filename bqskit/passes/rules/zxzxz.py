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

# Tolerance for recognizing the ZXZXZ middle rotation angle as Clifford (0 or
# +-pi). This only has to separate those two values from every other angle
# the decomposition produces, and both come out exact to floating-point
# precision, so this can be tight.
_CLIFFORD_ANGLE_TOL = 1e-10


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

        # When the middle rotation is Clifford (t == 0 or +-pi), the middle
        # SX.RZ(t).SX block is diagonal (or X) and commutes with the outer RZ
        # gates, so only l + p (or p - l) is actually determined by the
        # target unitary -- how that total is split between the two outer
        # gates is a free gauge choice. Splitting it evenly, as the formulas
        # above do unconditionally, turns one gate that needs synthesis into
        # two (e.g. every diagonal single-qubit unitary, which is common in
        # practice -- any run of Z-axis rotations reduces to one). Collapse
        # the gauge instead, so the whole rotation lands on one outer gate
        # and the other becomes a free identity (angle 0, exact in both RZ
        # and U1 parameterization).
        if abs(abs(t) - np.pi) < _CLIFFORD_ANGLE_TOL:
            l, p = 0.0, (l + p + np.pi) % (2 * np.pi) - np.pi
        elif abs(t) < _CLIFFORD_ANGLE_TOL:
            l, p = 0.0, (p - l + np.pi) % (2 * np.pi) - np.pi

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
