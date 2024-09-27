from __future__ import annotations

import numpy as np

from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized import VariableUnitaryGate
from bqskit.passes import BlockZXZPass, FullBlockZXZPass
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

class TestBZXZ:
    def test_small_qubit_bzxz(compiler: Compiler) -> None:
        circuit = create_random_unitary_circ(4)
        utry = circuit.get_unitary()
        bzxz = BlockZXZPass(min_qudit_size=2)
        circuit = compiler.compile(circuit, [bzxz])
        dist = circuit.get_unitary().get_distance_from(utry)
        assert dist <= 1e-5

    def test_full_bzxz_no_extract(compiler: Compiler) -> None:
        circuit = create_random_unitary_circ(5)
        utry = circuit.get_unitary()
        bzxz = FullBlockZXZPass(min_qudit_size=2, perform_scan=False,
                            perform_extract=False)
        circuit = compiler.compile(circuit, [bzxz])
        dist = circuit.get_unitary().get_distance_from(utry)
        print(dist)
        print(circuit.gate_counts)
        assert dist <= 1e-5