from __future__ import annotations

import numpy as np
import pytest
from bqskitrs import Circuit

from bqskit.ir import Circuit as Circ
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import RXGate
from bqskit.ir.gates import RYGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import U1Gate
from bqskit.ir.gates import U2Gate
from bqskit.ir.gates import U3Gate

NATIVE_GATES = (
    RXGate(),
    RYGate(),
    RZGate(),
    CNOTGate(),
    U1Gate(),
    U2Gate(),
    U3Gate(),
)


@pytest.mark.parametrize('gate', NATIVE_GATES, ids=lambda gate: repr(gate))
def test_get_unitary(gate: Gate) -> None:
    size = gate.get_size()
    circ = Circ(size)
    circ.append_gate(gate, list(range(size)))
    num_params = circ.get_num_params()
    x = np.random.random((num_params,))
    circuit = Circuit(circ)
    assert np.allclose(circ.get_unitary(x).get_numpy(), circuit.get_unitary(x))


@pytest.mark.parametrize('gate', NATIVE_GATES, ids=lambda gate: repr(gate))
def test_get_grad(gate: Gate) -> None:
    size = gate.get_size()
    circ = Circ(size)
    circ.append_gate(gate, list(range(size)))
    num_params = circ.get_num_params()
    x = np.random.random((num_params,))
    circuit = Circuit(circ)
    grad_python = circ.get_grad(x)
    grad_rust = circuit.get_grad(x)
    for py, rs in zip(grad_python, grad_rust):
        assert np.allclose(py, rs)


@pytest.mark.parametrize('gate', NATIVE_GATES, ids=lambda gate: repr(gate))
def test_get_unitary_and_grad(gate: Gate) -> None:
    size = gate.get_size()
    circ = Circ(size)
    circ.append_gate(gate, list(range(size)))
    num_params = circ.get_num_params()
    x = np.random.random((num_params,))
    circuit = Circuit(circ)
    utry_python, grad_python = circ.get_unitary_and_grad(x)
    utry_rust, grad_rust = circuit.get_unitary_and_grad(x)
    assert np.allclose(utry_python.get_numpy(), utry_rust)
    for i, (py, rs) in enumerate(zip(grad_python, grad_rust)):
        assert np.allclose(py, rs)
