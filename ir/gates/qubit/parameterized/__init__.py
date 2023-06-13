"""This package contains parameterized gates."""
from __future__ import annotations

from bqskit.ir.gates.qubit.parameterized.ccp import CCPGate
from bqskit.ir.gates.qubit.parameterized.cp import CPGate
from bqskit.ir.gates.qubit.parameterized.crx import CRXGate
from bqskit.ir.gates.qubit.parameterized.cry import CRYGate
from bqskit.ir.gates.qubit.parameterized.crz import CRZGate
from bqskit.ir.gates.qubit.parameterized.cu import CUGate
from bqskit.ir.gates.qubit.parameterized.fsim import FSIMGate
from bqskit.ir.gates.qubit.parameterized.pauli import PauliGate
from bqskit.ir.gates.qubit.parameterized.phasedxz import PhasedXZGate
from bqskit.ir.gates.qubit.parameterized.rx import RXGate
from bqskit.ir.gates.qubit.parameterized.rxx import RXXGate
from bqskit.ir.gates.qubit.parameterized.ry import RYGate
from bqskit.ir.gates.qubit.parameterized.ryy import RYYGate
from bqskit.ir.gates.qubit.parameterized.rz import RZGate
from bqskit.ir.gates.qubit.parameterized.rzz import RZZGate
from bqskit.ir.gates.qubit.parameterized.u1 import U1Gate
from bqskit.ir.gates.qubit.parameterized.u1q import U1qGate
from bqskit.ir.gates.qubit.parameterized.u1q import U1qPi2Gate
from bqskit.ir.gates.qubit.parameterized.u1q import U1qPiGate
from bqskit.ir.gates.qubit.parameterized.u2 import U2Gate
from bqskit.ir.gates.qubit.parameterized.u3 import U3Gate
from bqskit.ir.gates.qubit.parameterized.u8 import U8Gate
from bqskit.ir.gates.qubit.parameterized.unitary import VariableUnitaryGate

__all__ = [
    'CCPGate',
    'CPGate',
    'CRXGate',
    'CRYGate',
    'CRZGate',
    'CUGate',
    'FSIMGate',
    'PauliGate',
    'PhasedXZGate',
    'RXGate',
    'RXXGate',
    'RYGate',
    'RYYGate',
    'RZGate',
    'RZZGate',
    'U1Gate',
    'U1qGate',
    'U1qPiGate',
    'U1qPi2Gate',
    'U2Gate',
    'U3Gate',
    'U8Gate',
    'VariableUnitaryGate',
]
