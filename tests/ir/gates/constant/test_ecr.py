from __future__ import annotations

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import ECRGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import XGate
from bqskit.ir.lang.qasm2 import OPENQASM2Language


def test_ecr() -> None:
    c = Circuit(2)
    c.append_gate(HGate(), 1)
    c.append_gate(CXGate(), (0, 1))
    c.append_gate(RZGate(), 1, [np.pi / 4.0])
    c.append_gate(CXGate(), (0, 1))
    c.append_gate(HGate(), 1)

    c.append_gate(XGate(), 0)

    c.append_gate(HGate(), 1)
    c.append_gate(CXGate(), (0, 1))
    c.append_gate(RZGate(), 1, [-np.pi / 4.0])
    c.append_gate(CXGate(), (0, 1))
    c.append_gate(HGate(), 1)

    assert c.get_unitary().get_distance_from(ECRGate().get_unitary()) < 3e-8


def test_ecr_encode_decode() -> None:
    c = Circuit(2)
    c.append_gate(ECRGate(), (0, 1))

    output_qasm = c.to('qasm')

    resulting_circuit = OPENQASM2Language().decode(output_qasm)
    assert resulting_circuit[0, 0].gate == ECRGate()
