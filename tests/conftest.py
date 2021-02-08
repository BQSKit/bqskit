"""
BQSKit Tests Root conftest.py

This module defines several fixtures for use in this test suite.
"""

from bqskit.qis.unitarymatrix import UnitaryMatrix
import pytest
import numpy as np

from bqskit.ir.gates import *
from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate

from inspect import signature

BQSKIT_GATES = [ CNOTGate(),
                 CZGate(),
                 ISWAPGate(),
                 RXGate(),
                 RYGate(),
                 RZGate(),
                 SQRTCNOTGate(),
                 U1Gate(),
                 U2Gate(),
                 U3Gate(),
                 XGate(),
                 XXGate(),
                 YGate(),
                 ZGate(),
                 IdentityGate(1),
                 IdentityGate(2),
                 IdentityGate(3),
                 IdentityGate(4),
                 VariableUnitaryGate(),
                 SemiFixedParameterGate(),
                 FixedUnitaryGate(), ]

@pytest.fixture(params=BQSKIT_GATES, ids = lambda gate: repr(gate))
def gate(request) -> Gate:
    """Provides all of BQSKIT_GATES as a gate fixture."""
    return request.param


@pytest.fixture
def simple_circuit() -> Circuit:
    """Provides a simple circuit fixture."""
    circuit = Circuit(2)
    circuit.append_gate(XGate(), [0])
    circuit.append_gate(CNOTGate(), [0, 1])
    circuit.append_gate(XGate(), [1])
    circuit.append_gate(CNOTGate(), [1, 0])
    return circuit

@pytest.fixture
def swap_circuit() -> Circuit:
    """Provides a swap implemented with 3 cnots as a circuit fixture."""
    circuit = Circuit(2)
    circuit.append_gate(CNOTGate(), [0, 1])
    circuit.append_gate(CNOTGate(), [1, 0])
    circuit.append_gate(CNOTGate(), [0, 1])
    return circuit

TOFFOLI = np.asarray(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]] )

@pytest.fixture
def toffoli_unitary() -> UnitaryMatrix:
    return UnitaryMatrix( TOFFOLI )

