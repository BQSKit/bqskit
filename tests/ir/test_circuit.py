from __future__ import annotations

import pytest

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import XGate



def test_simple_circuit(simple_circuit: Circuit):
    assert simple_circuit.get_gate(0, 0) is XGate()
    assert simple_circuit.get_gate(0, 1) is None
    assert simple_circuit.get_gate(1, 0) is CNOTGate()
    assert simple_circuit.get_gate(1, 1) is CNOTGate()
    assert simple_circuit.get_gate(2, 0) is None
    assert simple_circuit.get_gate(2, 1) is XGate()
    assert simple_circuit.get_gate(3, 0) is CNOTGate()
    assert simple_circuit.get_gate(3, 1) is CNOTGate()
