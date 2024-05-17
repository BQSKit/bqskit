from __future__ import annotations

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import ECRGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import XGate

def test_ecr() -> None:
    c = Circuit(2)
    c.append_gate(HGate(), 1)
    c.append_gate(CXGate(), (0, 1))
    c.append_gate(RZGate(np.pi/4.0), 1)
    c.append_gate(CXGate(), (0, 1))
    c.append_gate(HGate(), 1)

    c.append_gate(XGate(), 0)

    c.append_gate(HGate(), 1)
    c.append_gate(CXGate(), (0, 1))
    c.append_gate(RZGate(-np.pi/4.0), 1)
    c.append_gate(CXGate(), (0, 1))
    c.append_gate(HGate(), 1)


    print(np.round(c.get_unitary(), 1))

    assert c.get_unitary().get_distance_from(ECRGate().get_unitary()) < 3e-8
