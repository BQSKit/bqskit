"""This package contains parameterized gates."""
from __future__ import annotations

from bqskit.ir.gates.parameterized.cp import CPGate
from bqskit.ir.gates.parameterized.crx import CRXGate
from bqskit.ir.gates.parameterized.cry import CRYGate
from bqskit.ir.gates.parameterized.crz import CRZGate
from bqskit.ir.gates.parameterized.fsim import FSIMGate
from bqskit.ir.gates.parameterized.pauli import PauliGate
from bqskit.ir.gates.parameterized.phasedxz import PhasedXZGate
from bqskit.ir.gates.parameterized.rx import RXGate
from bqskit.ir.gates.parameterized.rxx import RXXGate
from bqskit.ir.gates.parameterized.ry import RYGate
from bqskit.ir.gates.parameterized.ryy import RYYGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gates.parameterized.rzz import RZZGate
from bqskit.ir.gates.parameterized.u1 import U1Gate
from bqskit.ir.gates.parameterized.u1q import U1qGate
from bqskit.ir.gates.parameterized.u1q import U1qPi2Gate
from bqskit.ir.gates.parameterized.u1q import U1qPiGate
from bqskit.ir.gates.parameterized.u2 import U2Gate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.parameterized.u8 import U8Gate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.gates.parameterized.unitary_acc import VariableUnitaryGateAcc

__all__ = [
    'CPGate',
    'CRXGate',
    'CRYGate',
    'CRZGate',
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
    'VariableUnitaryGateAcc',
]
