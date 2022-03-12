from __future__ import annotations

from bqskit.exec.runners.sim import SimulationRunner
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import HGate


def test_sim() -> None:
    circuit = Circuit(2)
    circuit.append_gate(HGate(), 0)
    circuit.append_gate(CNOTGate(), (0, 1))
    results = SimulationRunner().run(circuit)
    counts = results.get_counts(1024)
    assert counts[0] == 512
    assert counts[1] == 0
    assert counts[2] == 0
    assert counts[3] == 512
