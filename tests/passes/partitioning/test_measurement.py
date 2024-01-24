from __future__ import annotations

from bqskit.compiler.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import MeasurementPlaceholder
from bqskit.ir.gates import CXGate
from bqskit.passes.partitioning.quick import QuickPartitioner


def test_barrier_stop_partitioning_across_some_circuit(
        compiler: Compiler,
) -> None:
    circuit = Circuit(4)
    circuit.append_gate(CXGate(), (0, 1))
    measurements = {0: ('c', 0), 1: ('c', 1)}
    circuit.append_gate(MeasurementPlaceholder([('c', 2)], measurements), (0, 1))
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(CXGate(), (2, 3))
    circuit.append_gate(CXGate(), (2, 3))
    circuit = compiler.compile(circuit, [QuickPartitioner(2)])
    assert circuit.num_operations == 4

