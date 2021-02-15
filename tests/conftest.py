"""
BQSKit Tests Root conftest.py.

This module defines several fixtures for use in this test suite. There
are three main types of fixtures defined here, unitaries, gates, and
circuits.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.stats import unitary_group

from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates import *
from bqskit.qis.unitarymatrix import UnitaryMatrix

# Unitaries
# Random and invalid unitaries dynamically generated in hooks below
TOFFOLI = np.asarray(
    [
        [
            1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
            0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
        ],
        [
            0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j,
            0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
        ],
        [
            0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j,
            0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
        ],
        [
            0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j,
            0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
        ],
        [
            0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
            1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
        ],
        [
            0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
            0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j,
        ],
        [
            0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
            0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j,
        ],
        [
            0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
            0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j,
        ],
    ],
)


@pytest.fixture
def toffoli_unitary() -> UnitaryMatrix:
    return UnitaryMatrix(TOFFOLI)


@pytest.fixture
def toffoli_unitary_np() -> UnitaryMatrix:
    return TOFFOLI


# Gates
BQSKIT_GATES = [
    CHGate(),
    CPIGate(),
    CSUMGate(),
    CXGate(),
    CNOTGate(),
    CYGate(),
    CZGate(),
    HGate(),
    IdentityGate(1),
    IdentityGate(2),
    IdentityGate(3),
    IdentityGate(4),
    ISwapGate(),
    # PermutationGate(),  # TODO
    SGate(),
    SdgGate(),
    SqrtCNOTGate(),
    SwapGate(),
    SXGate(),
    SqrtXGate(),
    TGate(),
    TdgGate(),
    ConstantUnitaryGate(TOFFOLI),  # TODO
    XGate(),
    XXGate(),
    YGate(),
    ZGate(),
    # PauliGate(),  # TODO
    RXGate(),
    RYGate(),
    RZGate(),
    U1Gate(),
    U2Gate(),
    U3Gate(),
    U8Gate(),
    DaggerGate(TGate()),
    DaggerGate(CZGate()),
    DaggerGate(U1Gate()),
    DaggerGate(U8Gate()),
    FrozenParameterGate(U1Gate(), {0: np.pi}),
    FrozenParameterGate(U3Gate(), {0: np.pi}),
    FrozenParameterGate(U3Gate(), {0: np.pi / 2, 1: np.pi / 2, 2: np.pi / 2}),
    FrozenParameterGate(U8Gate(), {0: np.pi}),
    FrozenParameterGate(U8Gate(), {0: np.pi / 2, 1: np.pi / 2, 2: np.pi / 2}),
    FrozenParameterGate(DaggerGate(U8Gate()), {0: np.pi / 2, 2: np.pi / 2}),
    # VariableUnitaryGate(TOFFOLI), # TODO
    # CircuitGate(),  # TODO
    # ControlledGate(),  # TODO
]

CONSTANT_GATES = [g for g in BQSKIT_GATES if g.is_constant()]
QUBIT_GATES = [g for g in BQSKIT_GATES if g.is_qubit_gate()]
QUTRIT_GATES = [g for g in BQSKIT_GATES if g.is_qutrit_gate()]
PARAMETERIZED_GATES = [g for g in BQSKIT_GATES if g.is_parameterized()]


@pytest.fixture(params=BQSKIT_GATES, ids=lambda gate: repr(gate))
def gate(request: Any) -> Gate:
    """Provides all of BQSKIT_GATES as a gate fixture."""
    return request.param


@pytest.fixture(params=CONSTANT_GATES, ids=lambda gate: repr(gate))
def constant_gate(request: Any) -> Gate:
    """Provides all of CONSTANT_GATES as a gate fixture."""
    return request.param


@pytest.fixture(params=QUBIT_GATES, ids=lambda gate: repr(gate))
def qubit_gate(request: Any) -> Gate:
    """Provides all of QUBIT_GATES as a gate fixture."""
    return request.param


@pytest.fixture(params=QUTRIT_GATES, ids=lambda gate: repr(gate))
def qutrit_gate(request: Any) -> Gate:
    """Provides all of QUTRIT_GATES as a gate fixture."""
    return request.param


@pytest.fixture(params=PARAMETERIZED_GATES, ids=lambda gate: repr(gate))
def param_gate(request: Any) -> Gate:
    """Provides all of PARAMETERIZED_GATES as a gate fixture."""
    return request.param


# Circuits
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

# Dynamically generated fixtures


def pytest_generate_tests(metafunc: Any) -> None:
    """
    Pytest Hook called when collecting test functions. Inject parameterized
    fixtures programmatically here.

    Used to generate random_unitary fixtures dynamically.
    If random_unitary is in the fixture name, this will parameterize
    that fixture automatically. Optional numbers seperated by underscores
    will provide dimension choices; additionally if 'np' (seperated by
    underscores) is in the fixture name, then the fixture will be provide
    ndarrays instead of UnitaryMatrix objects.

    Also used to generate invalid_unitary fixtures dynamically.
    If invalid_unitary is in the fixture name, this will parameterize
    that fixture automatically. Optional numbers seperated by underscores
    will provide dimension choices. This fixture will always return
    a numpy ndarray.

    Examples:
        def test_something(random_unitary_2):
            ... # random_unitary_2 = 2x2 UnitaryMatrix
        def test_something(random_unitary_np_2_3_4)
            ... # random_unitary_np_2_3_4 = 2x2, 3x3, or 4x4 unitary ndarray
        def test_something(random_unitary):
            ... # random_unitary = NxN UnitaryMatrix, where N in range(2,10)

        def test_something(invalid_unitary_2):
            ... # invalid_unitary_2 = 2x2 ndarray that is not a unitary
    """
    for fixturename in metafunc.fixturenames:
        if 'random_unitary' in fixturename:
            np.random.seed(21211411)  # Set random seed for reproducibility
            tokens = fixturename.split['_']
            numpy_flag = 'np' in tokens
            dimensions = [int(token) for token in tokens if token.isdigit()]
            if len(dimensions) == 0:
                dimensions = list(range(2, 10))
            dimensions = np.random.choice(dimensions, 20)
            fixture = [unitary_group.rvs(dimensions[i]) for i in range(20)]
            if not numpy_flag:
                fixture = [UnitaryMatrix(m) for m in fixture]
            metafunc.parameterize(fixturename, fixture)

        if 'invalid_unitary' in fixturename:
            np.random.seed(21211411)  # Set random seed for reproducibility
            tokens = fixturename.split['_']
            dimensions = [int(token) for token in tokens if token.isdigit()]
            if len(dimensions) == 0:
                dimensions = list(range(2, 10))
            dimensions = np.random.choice(dimensions, 20)
            fixture = [unitary_group.rvs(dimensions[i]) for i in range(20)]
            fixture = [m + np.identity(len(m)) for m in fixture]
            metafunc.parameterize(fixturename, fixture)
