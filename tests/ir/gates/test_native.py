from bqskit.ir.gates import (
    RXGate, RYGate, RZGate, CNOTGate, U1Gate, U2Gate, U3Gate
)
from bqskitrs import Circuit
from bqskit.ir import Circuit as Circ

import numpy as np

import pytest

GATES = (
    RXGate(),
    RYGate(),
    RZGate(),
    CNOTGate(),
    U1Gate(),
    U2Gate(),
    U3Gate(),
)


@pytest.mark.parametrize("gate", GATES, ids=lambda gate: repr(gate))
def test_get_unitary(gate):
    size = gate.get_size()
    circ = Circ(size)
    circ.append_gate(gate, list(range(size)))
    num_params = circ.get_num_params()
    x = np.random.random((num_params,))
    circuit = Circuit(circ)
    assert np.allclose(circ.get_unitary(x).get_numpy(), circuit.get_unitary(x))

@pytest.mark.parametrize("gate", GATES, ids=lambda gate: repr(gate))
def test_get_grad(gate):
    size = gate.get_size()
    circ = Circ(size)
    circ.append_gate(gate, list(range(size)))
    num_params = circ.get_num_params()
    x = np.random.random((num_params,))
    circuit = Circuit(circ)
    grad_python = circ.get_grad(x)
    grad_rust = circuit.get_grad(x)
    for py, rs in zip(grad_python, grad_rust):
        assert np.allclose(py, rs), print(py) or print(rs)

@pytest.mark.parametrize("gate", GATES, ids=lambda gate: repr(gate))
def test_get_unitary_and_grad(gate):
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
        assert np.allclose(py, rs), print(py) or print(rs) or i
