from __future__ import annotations

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import RC3XGate
from bqskit.ir.gates import RCCXGate
from bqskit.ir.gates import TdgGate
from bqskit.ir.gates import TGate


def test_margolus() -> None:
    c = Circuit(3)
    c.append_gate(HGate(), 2)
    c.append_gate(TGate(), 2)
    c.append_gate(CXGate(), (1, 2))
    c.append_gate(TdgGate(), 2)
    c.append_gate(CXGate(), (0, 2))
    c.append_gate(TGate(), 2)
    c.append_gate(CXGate(), (1, 2))
    c.append_gate(TdgGate(), 2)
    c.append_gate(HGate(), 2)
    assert c.get_unitary().get_distance_from(RCCXGate().get_unitary()) < 1e-8


def test_rc3x() -> None:
    c = Circuit(4)
    c.append_gate(HGate(), 3)
    c.append_gate(TGate(), 3)
    c.append_gate(CXGate(), (2, 3))
    c.append_gate(TdgGate(), 3)
    c.append_gate(HGate(), 3)
    c.append_gate(CXGate(), (0, 3))
    c.append_gate(TGate(), 3)
    c.append_gate(CXGate(), (1, 3))
    c.append_gate(TdgGate(), 3)
    c.append_gate(CXGate(), (0, 3))
    c.append_gate(TGate(), 3)
    c.append_gate(CXGate(), (1, 3))
    c.append_gate(TdgGate(), 3)
    c.append_gate(HGate(), 3)
    c.append_gate(TGate(), 3)
    c.append_gate(CXGate(), (2, 3))
    c.append_gate(TdgGate(), 3)
    c.append_gate(HGate(), 3)
    import numpy as np
    print(np.round(c.get_unitary(), 1))

    assert c.get_unitary().get_distance_from(RC3XGate().get_unitary()) < 1e-8
