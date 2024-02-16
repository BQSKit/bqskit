from __future__ import annotations

from bqskit.compiler.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import MeasurementPlaceholder
from bqskit.passes.partitioning.quick import QuickPartitioner


def test_measurement_stop_partitioning_across_some_circuit(
        compiler: Compiler,
) -> None:
    circuit = Circuit(4)
    circuit.append_gate(CXGate(), (0, 1))
    measurements = {0: ('c', 0), 1: ('c', 1)}
    circuit.append_gate(
        MeasurementPlaceholder(
            [('c', 2)], measurements,
        ), (0, 1),
    )
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(CXGate(), (2, 3))
    circuit.append_gate(CXGate(), (2, 3))
    output_circuit = compiler.compile(circuit, [QuickPartitioner(2)])
    measurements = {0: ('c', 0), 1: ('c', 1)}
    measure = MeasurementPlaceholder([('c', 2)], measurements)
    assert output_circuit.num_operations == 4
    assert all(
        isinstance(op.gate, (CircuitGate, MeasurementPlaceholder))
        for op in output_circuit
    )
    assert isinstance(output_circuit[0, 0].gate, CircuitGate)
    assert output_circuit[0, 0].gate._circuit.num_qudits == 2
    assert output_circuit[0, 0].gate._circuit.gate_counts[CXGate()] == 1
    assert output_circuit[1, 0].gate == measure
    assert isinstance(output_circuit[2, 0].gate, CircuitGate)
    assert output_circuit[2, 0].gate._circuit.num_qudits == 2
    assert output_circuit[2, 0].gate._circuit.gate_counts[CXGate()] == 1
    assert isinstance(output_circuit[0, 2].gate, CircuitGate)
    assert output_circuit[0, 2].gate._circuit.num_qudits == 2
    assert output_circuit[0, 2].gate._circuit.gate_counts[CXGate()] == 2
