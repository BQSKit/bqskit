from __future__ import annotations

from bqskit.exec.runners.sim import SimulationRunner
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import HGate

circuit = Circuit(2)
circuit.append_gate(HGate(), 0)
circuit.append_gate(CNOTGate(), (0, 1))
results = SimulationRunner().run(circuit)
print(results.get_counts(1024))
