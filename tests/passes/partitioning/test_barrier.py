from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CXGate, U3Gate, CircuitGate
from bqskit.ir.gates import BarrierPlaceholder
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit import compile


def test_barrier_still_there_after_partitioning() -> None:
    circuit = Circuit(2)
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(BarrierPlaceholder(2), (0, 1))
    circuit.append_gate(CXGate(), (0, 1))
    circuit.perform(QuickPartitioner(3))
    assert circuit.num_operations == 3
    assert isinstance(circuit[0, 0].gate, CircuitGate)
    assert isinstance(circuit[1, 0].gate, BarrierPlaceholder)
    assert isinstance(circuit[2, 0].gate, CircuitGate)
    circuit.unfold_all()
    assert isinstance(circuit[0, 0].gate, CXGate)
    assert isinstance(circuit[1, 0].gate, BarrierPlaceholder)
    assert isinstance(circuit[2, 0].gate, CXGate)


def test_barrier_stop_partitioning_across_some_circuit() -> None:
    circuit = Circuit(4)
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(BarrierPlaceholder(2), (0, 1))
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(CXGate(), (2, 3))
    circuit.append_gate(CXGate(), (2, 3))
    circuit.perform(QuickPartitioner(2))
    assert circuit.num_operations == 4


def test_barrier_end_to_end_single_qubit_syn() -> None:
    circuit = Circuit(1)
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(BarrierPlaceholder(1), 0)
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(BarrierPlaceholder(1), 0)
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(BarrierPlaceholder(1), 0)
    circuit.append_gate(U3Gate(), 0)
    out = compile(circuit)
    assert out.num_operations == 7
