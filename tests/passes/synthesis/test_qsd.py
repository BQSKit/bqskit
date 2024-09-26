from __future__ import annotations

import numpy as np

from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized import VariableUnitaryGate
from bqskit.passes import FullQSDPass
from bqskit.qis import UnitaryMatrix


def create_random_unitary_circ(num_qudits: int):
    '''
    Create a Circuit with a random VariableUnitaryGate.
    '''
    circuit = Circuit(num_qudits)
    utry = UnitaryMatrix.random(num_qudits)
    utry_params = np.concatenate((np.real(utry._utry).flatten(),
                                  np.imag(utry._utry).flatten()))
    circuit.append_gate(VariableUnitaryGate(num_qudits),
                        list(range(num_qudits)),
                        utry_params)
    return circuit

class TestQSD:
    def test_three_qubit_qsd(self, compiler: Compiler) -> None:
        circuit = create_random_unitary_circ(3)
        utry = circuit.get_unitary()
        # Run one pass of QSD
        qsd = FullQSDPass(min_qudit_size=2, perform_scan=False)
        circuit = compiler.compile(circuit, [qsd])
        dist = circuit.get_unitary().get_distance_from(utry)
        assert circuit.count(VariableUnitaryGate(2)) == 4
        assert circuit.count(VariableUnitaryGate(3)) == 0
        assert dist <= 1e-5

    def test_four_qubit_qubit(self, compiler: Compiler) -> None:
        circuit = create_random_unitary_circ(4)
        utry = circuit.get_unitary()
        # Run two passes of QSD
        qsd = FullQSDPass(min_qudit_size=2, perform_scan=False)
        circuit = compiler.compile(circuit, [qsd])
        dist = circuit.get_unitary().get_distance_from(utry)
        assert circuit.count(VariableUnitaryGate(2)) == 16
        assert circuit.count(VariableUnitaryGate(3)) == 0
        assert circuit.count(VariableUnitaryGate(4)) == 0
        assert dist <= 1e-5

    def test_five_qubit_qsd(self, compiler: Compiler) -> None:
        circuit = create_random_unitary_circ(5)
        utry = circuit.get_unitary()
        # Run two passes of QSD
        qsd = FullQSDPass(min_qudit_size=3, perform_scan=False)
        circuit = compiler.compile(circuit, [qsd])
        dist = circuit.get_unitary().get_distance_from(utry)
        assert circuit.count(VariableUnitaryGate(3)) == 16
        assert circuit.count(VariableUnitaryGate(4)) == 0
        assert circuit.count(VariableUnitaryGate(5)) == 0
        assert dist <= 1e-5