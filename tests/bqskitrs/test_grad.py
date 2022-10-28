from __future__ import annotations

import numpy as np
from bqskitrs import Circuit as CircuitRS

from bqskit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import U3Gate


def test_basic() -> None:
    circuit = Circuit(3)
    for _ in range(10):
        for i in range(2):
            circuit.append_gate(CNOTGate(), (i, i + 1))
            circuit.append_gate(U3Gate(), i)
            circuit.append_gate(U3Gate(), i + 1)
    crs = CircuitRS(circuit)
    d, dM = crs.get_unitary_and_grad([0] * (2 * 10 * 6))
    od, odM = circuit.get_unitary_and_grad([0] * (2 * 10 * 6))
    assert np.allclose(d, od)
    assert np.allclose(dM, odM)
