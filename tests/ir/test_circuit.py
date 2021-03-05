from __future__ import annotations

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import XGate


def test_simple_circuit(simple_circuit: Circuit) -> None:
    assert simple_circuit[0, 0].gate is XGate()
    assert simple_circuit[0, 1].gate is None
    assert simple_circuit[1, 0].gate is CNOTGate()
    assert simple_circuit[1, 1].gate is CNOTGate()
    assert simple_circuit[2, 0].gate is None
    assert simple_circuit[2, 1].gate is XGate()
    assert simple_circuit[3, 0].gate is CNOTGate()
    assert simple_circuit[3, 1].gate is CNOTGate()
