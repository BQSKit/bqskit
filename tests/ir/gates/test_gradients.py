from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.circuit import Circuit
from bqskitrs import Circuit as Circ

import numpy as np

def check_gradient(circ: Circuit, num_params: int) -> None:
    totaldiff = [0] * num_params
    eps = 1e-5
    repeats = 100
    for _ in range(repeats):
        v = np.random.rand(num_params) * 2 * np.pi
        M, Js = circ.get_unitary_and_grad(v)
        for i in range(num_params):
            v2 = np.copy(v)
            v2[i] = v[i] + eps
            U1 = circ.get_unitary(v2)
            if isinstance(U1, UnitaryMatrix):
                U1 = U1.get_numpy()
            v2[i] = v[i] - eps
            U2 = circ.get_unitary(v2)
            if isinstance(U2, UnitaryMatrix):
                U2 = U2.get_numpy()

            FD = (U1 - U2) / (2*eps)

            diffs = np.sum(np.abs(FD - Js[i]))
            totaldiff[i] += diffs
    for i in range(num_params):
        assert totaldiff[i] < eps


def test_gradients(r3_qubit_circuit: Circuit) -> None:
    check_gradient(r3_qubit_circuit, r3_qubit_circuit.get_num_params())


def test_gradients_native(r3_qubit_circuit: Circuit) -> None:
    circ = Circ(r3_qubit_circuit)
    check_gradient(circ, r3_qubit_circuit.get_num_params())