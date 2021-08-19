from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from bqskitrs import Circuit

from bqskit.ir import Circuit as Circ
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import RXGate
from bqskit.ir.gates import RXXGate
from bqskit.ir.gates import RYGate
from bqskit.ir.gates import RYYGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import RZZGate
from bqskit.ir.gates import U1Gate
from bqskit.ir.gates import U2Gate
from bqskit.ir.gates import U3Gate
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.ir.gates.parameterized.pauli import PauliGate


NATIVE_GATES: list[Gate] = [
    RXGate(),
    RYGate(),
    RZGate(),
    RXXGate(),
    RYYGate(),
    RZZGate(),
    CNOTGate(),
    U1Gate(),
    U2Gate(),
    U3Gate(),
    PauliGate(1),
    PauliGate(2),
    PauliGate(3),
]


NON_GRADIENT_GATES: list[Gate] = [
    VariableUnitaryGate(2),
    VariableUnitaryGate(3),
    VariableUnitaryGate(4),
]


@pytest.mark.parametrize(
    'gate', NATIVE_GATES + NON_GRADIENT_GATES,
    ids=lambda gate: repr(gate),
)
def test_get_unitary(gate: Gate) -> None:
    size = gate.get_size()
    circ = Circ(size, radixes=gate.get_radixes())
    circ.append_gate(gate, list(range(size)))
    num_params = circ.get_num_params()
    x = np.random.random((num_params,))
    circuit = Circuit(circ)
    assert np.allclose(circ.get_unitary(x), circuit.get_unitary(x))


@pytest.mark.parametrize('gate', NATIVE_GATES, ids=lambda gate: repr(gate))
def test_get_grad(gate: Gate) -> None:
    size = gate.get_size()
    circ = Circ(size, radixes=gate.get_radixes())
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
    circ = Circ(size, radixes=gate.get_radixes())
    circ.append_gate(gate, list(range(size)))
    num_params = circ.get_num_params()
    x = np.random.random((num_params,))
    circuit = Circuit(circ)
    utry_python, grad_python = circ.get_unitary_and_grad(x)
    utry_rust, grad_rust = circuit.get_unitary_and_grad(x)
    assert np.allclose(utry_python, utry_rust)
    for i, (py, rs) in enumerate(zip(grad_python, grad_rust)):
        assert np.allclose(py, rs)


def test_random_circuit_only_native(
        gen_random_circuit: Any,
) -> None:
    circ = gen_random_circuit(3, gateset=NATIVE_GATES + NON_GRADIENT_GATES)
    num_params = circ.get_num_params()
    x = np.random.random((num_params,))
    circuit = Circuit(circ)
    assert np.allclose(
        circ.get_unitary(x),
        circuit.get_unitary(x),
    ), circ._circuit


def test_random_circuit_qubit_gates(qubit_gate: Gate) -> None:
    circ = Circ(qubit_gate.get_size())
    circ.append_gate(qubit_gate, location=list(range(qubit_gate.get_size())))
    num_params = circ.get_num_params()
    if qubit_gate.is_constant():
        assert num_params == 0
    x = np.random.random((num_params,))
    circuit = Circuit(circ)
    py = circ.get_unitary(x)
    rs = circuit.get_unitary(x)
    assert py.shape == rs.shape
    assert py.dtype is rs.dtype
    np.testing.assert_allclose(py, rs, verbose=True)


def test_random_circuit_qutrit_gates(qutrit_gate: Gate) -> None:
    size = qutrit_gate.get_size()
    circ = Circ(size, radixes=[3] * size)
    circ.append_gate(qutrit_gate, location=list(range(size)))
    num_params = circ.get_num_params()
    if qutrit_gate.is_constant():
        assert num_params == 0
    x = np.random.random((num_params,))
    circuit = Circuit(circ)
    py = circ.get_unitary(x)
    rs = circuit.get_unitary(x)
    assert py.shape == rs.shape
    assert py.dtype is rs.dtype
    np.testing.assert_allclose(py, rs, verbose=True)
