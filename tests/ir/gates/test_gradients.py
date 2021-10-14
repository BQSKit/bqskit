from __future__ import annotations

import numpy as np
from bqskitrs import Circuit as Circ

from bqskit.ir.circuit import Circuit
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


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
                utry1 = U1
            else:
                utry1 = U1
            v2[i] = v[i] - eps
            U2 = circ.get_unitary(v2)
            if isinstance(U2, UnitaryMatrix):
                utry2 = U2
            else:
                utry2 = U2

            FD = (utry1 - utry2) / (2 * eps)

            diffs = np.sum(np.abs(FD - Js[i]))
            totaldiff[i] += diffs
    for i in range(num_params):
        assert totaldiff[i] < eps


def test_gradients(r3_qubit_circuit: Circuit) -> None:
    check_gradient(r3_qubit_circuit, r3_qubit_circuit.num_params)


def test_gradients_native(r3_qubit_circuit: Circuit) -> None:
    circ = Circ(r3_qubit_circuit)
    check_gradient(circ, r3_qubit_circuit.num_params)
