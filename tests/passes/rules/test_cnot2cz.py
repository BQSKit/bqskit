from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import CZGate
from bqskit.ir.gates import U3Gate
from bqskit.passes import CNOTToCZPass


def test_cnot2cz_only_cnots() -> None:
    circuit = Circuit(3)
    for i in range(100):
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(CNOTGate(), (1, 2))

    utry = circuit.get_unitary()    
    CNOTToCZPass().run(circuit)
    assert CNOTGate() not in circuit.gate_set
    assert CZGate() in circuit.gate_set
    assert circuit.get_unitary().get_distance_from(utry) < 5e-8


def test_cnot2cz_with_single_qubit() -> None:
    circuit = Circuit(3)
    for i in range(100):
        circuit.append_gate(U3Gate(), 0)
        circuit.append_gate(U3Gate(), 1)
        circuit.append_gate(U3Gate(), 2)
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(U3Gate(), 0)
        circuit.append_gate(U3Gate(), 1)
        circuit.append_gate(CNOTGate(), (1, 2))
        circuit.append_gate(U3Gate(), 0)
        circuit.append_gate(U3Gate(), 1)
        circuit.append_gate(U3Gate(), 2)

    utry = circuit.get_unitary()    
    CNOTToCZPass().run(circuit)
    assert CNOTGate() not in circuit.gate_set
    assert CZGate() in circuit.gate_set
    assert circuit.get_unitary().get_distance_from(utry) < 5e-8
