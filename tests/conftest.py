"""
BQSKit Tests Root conftest.py.

This module defines several fixtures for use in this test suite. There are three
main types of fixtures defined here, unitaries, gates, and circuits.
"""
from __future__ import annotations

import os
from typing import Any
from typing import Callable
from typing import Sequence

import numpy as np
import pytest
from scipy.stats import unitary_group

from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import ConstantUnitaryGate
from bqskit.ir.gates import CPIGate
from bqskit.ir.gates import CSUMGate
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import CYGate
from bqskit.ir.gates import CZGate
from bqskit.ir.gates import DaggerGate
from bqskit.ir.gates import FrozenParameterGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import IdentityGate
from bqskit.ir.gates import ISwapGate
from bqskit.ir.gates import RXGate
from bqskit.ir.gates import RYGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import SdgGate
from bqskit.ir.gates import SGate
from bqskit.ir.gates import SqrtCNOTGate
from bqskit.ir.gates import SqrtXGate
from bqskit.ir.gates import SwapGate
from bqskit.ir.gates import SXGate
from bqskit.ir.gates import TdgGate
from bqskit.ir.gates import TGate
from bqskit.ir.gates import U1Gate
from bqskit.ir.gates import U2Gate
from bqskit.ir.gates import U3Gate
from bqskit.ir.gates import U8Gate
from bqskit.ir.gates import XGate
from bqskit.ir.gates import XXGate
from bqskit.ir.gates import YGate
from bqskit.ir.gates import ZGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_sequence
# from bqskit.ir.gates import CircuitGate
# from bqskit.ir.gates import ControlledGate
# from bqskit.ir.gates import PauliGate
# from bqskit.ir.gates import PermutationGate
# from bqskit.ir.gates import VariableUnitaryGate

# region Test Variables

NUMBER_RANDOM_CIRCUITS = int(
    os.environ['BQSKIT_PYTEST_NUM_RCIRCUITS']
    if 'BQSKIT_PYTEST_NUM_RCIRCUITS' in os.environ
    else 10,
)

# endregion

# region Types

type_dict = {
    'str_1': '',
    'str_2': 'a',
    'str_3': 'abc',
    'str_4': "abc@!@$$&%^()!@*^<>[]mM,./_+-=\'\"",
    'int_1': -1,
    'int_2': 0,
    'int_3': 1,
    'int_4': -127649217,
    'int_5': 212121212,
    'int_6': np.byte(1234),
    'int_7': np.short(1234),
    'int_8': np.intc(1234),
    'int_9': np.longlong(1234),
    'int_10': np.int8(1234),
    'int_11': np.int16(1234),
    'int_12': np.int32(1234),
    'int_13': np.int64(1234),
    'float_1': 0.0,
    'float_2': 1.0,
    'float_3': 1e-15,
    'float_4': 1.26738e14,
    'float_5': 1.23,
    'float_6': np.half(1234.0),
    'float_7': np.single(1234.0),
    'float_8': np.double(1234.0),
    'float_9': np.longdouble(1234.0),
    'float_10': np.float32(1234.0),
    'float_11': np.float64(1234.0),
    # Needs typeshed sync
    'complex_1': complex(0.0j),  # type: ignore
    # Needs typeshed sync
    'complex_2': complex(0.0 + 0.0j),    # type: ignore
    'complex_3': 1.0j,
    'complex_4': 1.0 + 1.0j,
    'complex_5': 1.0 - 1.0j,
    'complex_7': np.csingle(1234 + 1234j),
    'complex_8': np.cdouble(1234 + 1234j),
    'complex_9': np.clongdouble(1234 + 1234j),
    # needs newer numpy
    'complex_10': np.complex64(1234 + 1234j),  # type: ignore
    # needs newer numpy
    'complex_11': np.complex128(1234 + 1234j),  # type: ignore
    'bool_1': False,
    'bool_2': True,
    'bool_3': np.bool_(False),
    'bool_4': np.bool_(True),
    'seq-str_1': [],
    'seq-str_2': ['0'],
    'seq-str_3': ['0, 1', 'abc', '@#!$^%&(#'],
    'seq-str_4': ['A'] * 10,
    'seq-int_1': [],
    'seq-int_2': [0],
    'seq-int_3': [0, 1, np.int8(1), np.int16(1), np.int32(1), np.int64(1)],
    'seq-int_4': [3] * 10,
    'seq-float_1': [],
    'seq-float_2': [0.0],
    'seq-float_3': [0.0, 1.23, 1e-14, np.float32(1.0), np.float64(1.0)],
    'seq-float_4': [1.234e12] * 10,
    'seq-complex_1': [],
    'seq-complex_2': [0.0j],

    'seq-complex_3': [
        0.0j,
        1.23j,
        1.1 + 1.0j,
        np.complex64(1.0 + 1.0j),  # type: ignore # needs newer numpy
    ],
    'seq-complex_4': [1.234e12j] * 10,
    'seq-bool_1': [],
    'seq-bool_2': [False],
    'seq-bool_3': [True],
    'seq-bool_4': [True, False, True, False, np.bool_(False), np.bool_(True)],
    'seq-bool_5': [False] * 10,
    'seq-bool_6': [True] * 10,
}

# str


@pytest.fixture(
    params=[(k, v) for k, v in type_dict.items() if k.split('_')[0] == 'str'],
    ids=lambda tup: tup[0],
)
def a_str(request: Any) -> str:
    """Provide random values that are strs."""
    return request.param[1]


@pytest.fixture(
    params=[(k, v) for k, v in type_dict.items() if k.split('_')[0] != 'str'],
    ids=lambda tup: tup[0],
)
def not_a_str(request: Any) -> Any:
    """Provide random values that are not strs."""
    return request.param[1]


# int
@pytest.fixture(
    params=[(k, v) for k, v in type_dict.items() if k.split('_')[0] == 'int'],
    ids=lambda tup: tup[0],
)
def an_int(request: Any) -> int:
    """Provide random values that are ints."""
    return request.param[1]


@pytest.fixture(
    params=[(k, v) for k, v in type_dict.items() if k.split('_')[0] != 'int'],
    ids=lambda tup: tup[0],
)
def not_an_int(request: Any) -> Any:
    """Provide random values that are not ints."""
    return request.param[1]


# float
@pytest.fixture(
    params=[
        (k, v) for k, v in type_dict.items()
        if k.split('_')[0] == 'float'
    ],
    ids=lambda tup: tup[0],
)
def a_float(request: Any) -> float:
    """Provide random values that are floats."""
    return request.param[1]


@pytest.fixture(
    params=[
        (k, v) for k, v in type_dict.items()
        if k.split('_')[0] != 'float'
    ],
    ids=lambda tup: tup[0],
)
def not_a_float(request: Any) -> Any:
    """Provide random values that are not floats."""
    return request.param[1]


# complex
@pytest.fixture(
    params=[
        (k, v) for k, v in type_dict.items()
        if k.split('_')[0] == 'complex'
    ],
    ids=lambda tup: tup[0],
)
def a_complex(request: Any) -> complex:
    """Provide random values that are complexs."""
    return request.param[1]


@pytest.fixture(
    params=[
        (k, v) for k, v in type_dict.items()
        if k.split('_')[0] != 'complex'
    ],
    ids=lambda tup: tup[0],
)
def not_a_complex(request: Any) -> Any:
    """Provide random values that are not complexs."""
    return request.param[1]


# bool
@pytest.fixture(
    params=[(k, v) for k, v in type_dict.items() if k.split('_')[0] == 'bool'],
    ids=lambda tup: tup[0],
)
def a_bool(request: Any) -> bool:
    """Provide random values that are bools."""
    return request.param[1]


@pytest.fixture(
    params=[(k, v) for k, v in type_dict.items() if k.split('_')[0] != 'bool'],
    ids=lambda tup: tup[0],
)
def not_a_bool(request: Any) -> Any:
    """Provide random values that are not bools."""
    return request.param[1]


# seq-str
@pytest.fixture(
    params=[
        (k, v) for k, v in type_dict.items()
        if k.split('_')[0] == 'seq-str'
    ],
    ids=lambda tup: tup[0],
)
def a_seq_str(request: Any) -> Sequence[str]:
    """Provide random values that are sequences of strs."""
    return request.param[1]


@pytest.fixture(
    params=[
        (k, v) for k, v in type_dict.items()
        if k.split('_')[0] != 'seq-str'
        and (not isinstance(v, list) or len(v) != 0)
    ],
    ids=lambda tup: tup[0],
)
def not_a_seq_str(request: Any) -> Any:
    """Provide random values that are not sequences of strs."""
    return request.param[1]


# seq-int
@pytest.fixture(
    params=[
        (k, v) for k, v in type_dict.items()
        if k.split('_')[0] == 'seq-int'
    ],
    ids=lambda tup: tup[0],
)
def a_seq_int(request: Any) -> Sequence[int]:
    """Provide random values that are sequences of ints."""
    return request.param[1]


@pytest.fixture(
    params=[
        (k, v) for k, v in type_dict.items()
        if k.split('_')[0] != 'seq-int'
        and (not isinstance(v, list) or len(v) != 0)
    ],
    ids=lambda tup: tup[0],
)
def not_a_seq_int(request: Any) -> Any:
    """Provide random values that are not sequences of ints."""
    return request.param[1]


# seq-float
@pytest.fixture(
    params=[
        (k, v) for k, v in type_dict.items()
        if k.split('_')[0] == 'seq-float'
    ],
    ids=lambda tup: tup[0],
)
def a_seq_float(request: Any) -> Sequence[float]:
    """Provide random values that are sequences of floats."""
    return request.param[1]


@pytest.fixture(
    params=[
        (k, v) for k, v in type_dict.items()
        if k.split('_')[0] != 'seq-float'
        and (not isinstance(v, list) or len(v) != 0)
    ],
    ids=lambda tup: tup[0],
)
def not_a_seq_float(request: Any) -> Any:
    """Provide random values that are not sequences of floats."""
    return request.param[1]


# seq-complex
@pytest.fixture(
    params=[
        (k, v) for k, v in type_dict.items()
        if k.split('_')[0] == 'seq-complex'
    ],
    ids=lambda tup: tup[0],
)
def a_seq_complex(request: Any) -> Sequence[complex]:
    """Provide random values that are sequences of complexs."""
    return request.param[1]


@pytest.fixture(
    params=[
        (k, v) for k, v in type_dict.items()
        if k.split('_')[0] != 'seq-complex'
        and (not isinstance(v, list) or len(v) != 0)
    ],
    ids=lambda tup: tup[0],
)
def not_a_seq_complex(request: Any) -> Any:
    """Provide random values that are not sequences of complexs."""
    return request.param[1]


# seq-bool
@pytest.fixture(
    params=[
        (k, v) for k, v in type_dict.items()
        if k.split('_')[0] == 'seq-bool'
    ],
    ids=lambda tup: tup[0],
)
def a_seq_bool(request: Any) -> Sequence[bool]:
    """Provide random values that are sequences of bools."""
    return request.param[1]


@pytest.fixture(
    params=[
        (k, v) for k, v in type_dict.items()
        if k.split('_')[0] != 'seq-bool'
        and (not isinstance(v, list) or len(v) != 0)
    ],
    ids=lambda tup: tup[0],
)
def not_a_seq_bool(request: Any) -> Any:
    """Provide random values that are not sequences of bools."""
    return request.param[1]


# endregion

# region Unitaries

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

SWAP = np.asarray(
    [
        [1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
        [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
        [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
        [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j],
    ],
)


@pytest.fixture
def toffoli_unitary() -> UnitaryMatrix:
    return UnitaryMatrix(TOFFOLI)


@pytest.fixture
def toffoli_unitary_np() -> np.ndarray:
    return TOFFOLI


@pytest.fixture
def swap_unitary() -> UnitaryMatrix:
    return UnitaryMatrix(SWAP)


@pytest.fixture
def swap_unitary_np() -> np.ndarray:
    return SWAP


def random_utry_gen(dims: int | Sequence[int]) -> np.ndarray:
    """
    Generate a random unitary.

    dims (int | Sequence[int]): The dimension of the matrix.
        If multiple are given then the dimension is randomly chosen
        from the ones given.

    Returns:
        (np.ndarray): The randomly generated unitary matrix.
    """
    if isinstance(dims, int):
        return unitary_group.rvs(dims)

    elif is_sequence(dims):
        if not all(isinstance(dim, int) for dim in dims):
            raise TypeError('Expected integer dimension.')

        return unitary_group.rvs(np.random.choice(dims, 1)[0])

    else:
        raise TypeError(
            'Expected integer or sequence of integers'
            ' for dimension, got: %s' % type(dims),
        )


@pytest.fixture
def gen_random_utry_np() -> Callable[[int | Sequence[int]], np.ndarray]:
    """Provide a method to generate random unitaries."""
    return random_utry_gen


def invalid_utry_gen(dims: int | Sequence[int]) -> np.ndarray:
    """
    Generate a random invalid unitary.

    dims (int | Sequence[int]): The dimension of the matrix. If multiple
        are given then the dimension is randomly chosen from the ones given.

    Returns:
        (np.ndarray): The randomly generated invalid unitary matrix.
    """
    if isinstance(dims, int):
        return unitary_group.rvs(dims) + np.identity(dims)

    elif is_sequence(dims):
        if not all(isinstance(dim, int) for dim in dims):
            raise TypeError('Expected integer dimension.')

        dim = np.random.choice(dims, 1)[0]
        return unitary_group.rvs(dim) + np.identity(dim)

    else:
        raise TypeError(
            'Expected integer or sequence of integers'
            ' for dimension, got: %s' % type(dims),
        )


@pytest.fixture
def gen_invalid_utry_np() -> Callable[[int | Sequence[int]], np.ndarray]:
    """Provide a method to generate random invalid unitaries."""
    return invalid_utry_gen


# endregion

# region Gates

BQSKIT_GATES = [
    CXGate(),
    CPIGate(),
    CSUMGate(),
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
QUBIT_GATES = [g for g in BQSKIT_GATES if g.is_qubit_only()]
QUTRIT_GATES = [g for g in BQSKIT_GATES if g.is_qutrit_only()]
PARAMETERIZED_GATES = [g for g in BQSKIT_GATES if g.is_parameterized()]
SINGLE_QUBIT_GATES = [
    g for g in BQSKIT_GATES
    if g.is_qubit_only() and g.num_qudits == 1
]
SINGLE_QUTRIT_GATES = [
    g for g in BQSKIT_GATES
    if g.is_qutrit_only() and g.num_qudits == 1
]
TWO_QUBIT_GATES = [
    g for g in BQSKIT_GATES
    if g.is_qubit_only() and g.num_qudits == 2
]
TWO_QUTRIT_GATES = [
    g for g in BQSKIT_GATES
    if g.is_qutrit_only() and g.num_qudits == 2
]
MULTI_QUBIT_GATES = [
    g for g in BQSKIT_GATES
    if g.is_qubit_only() and g.num_qudits >= 2
]
MULTI_QUTRIT_GATES = [
    g for g in BQSKIT_GATES
    if g.is_qutrit_only() and g.num_qudits >= 2
]


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


@pytest.fixture(params=SINGLE_QUBIT_GATES, ids=lambda gate: repr(gate))
def single_qubit_gate(request: Any) -> Gate:
    """Provides all of SINGLE_QUBIT_GATES as a gate fixture."""
    return request.param


@pytest.fixture(params=SINGLE_QUTRIT_GATES, ids=lambda gate: repr(gate))
def single_qutrit_gate(request: Any) -> Gate:
    """Provides all of SINGLE_QUTRIT_GATES as a gate fixture."""
    return request.param


@pytest.fixture(params=TWO_QUBIT_GATES, ids=lambda gate: repr(gate))
def two_qubit_gate(request: Any) -> Gate:
    """Provides all of two_QUBIT_GATES as a gate fixture."""
    return request.param


@pytest.fixture(params=TWO_QUTRIT_GATES, ids=lambda gate: repr(gate))
def two_qutrit_gate(request: Any) -> Gate:
    """Provides all of two_QUTRIT_GATES as a gate fixture."""
    return request.param


@pytest.fixture(params=MULTI_QUBIT_GATES, ids=lambda gate: repr(gate))
def multi_qubit_gate(request: Any) -> Gate:
    """Provides all of MULTI_QUBIT_GATES as a gate fixture."""
    return request.param


@pytest.fixture(params=MULTI_QUTRIT_GATES, ids=lambda gate: repr(gate))
def multi_qutrit_gate(request: Any) -> Gate:
    """Provides all of MULTI_QUTRIT_GATES as a gate fixture."""
    return request.param

# endregion

# region Circuits


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


@pytest.fixture
def toffoli_circuit() -> Circuit:
    """Provides a standard toffoli implemention."""
    circuit = Circuit(3)
    circuit.append_gate(HGate(), [2])
    circuit.append_gate(CNOTGate(), [1, 2])
    circuit.append_gate(TdgGate(), [2])
    circuit.append_gate(CNOTGate(), [0, 2])
    circuit.append_gate(TGate(), [2])
    circuit.append_gate(CNOTGate(), [1, 2])
    circuit.append_gate(TdgGate(), [2])
    circuit.append_gate(CNOTGate(), [0, 2])
    circuit.append_gate(TGate(), [1])
    circuit.append_gate(TGate(), [2])
    circuit.append_gate(CNOTGate(), [0, 1])
    circuit.append_gate(HGate(), [2])
    circuit.append_gate(TGate(), [0])
    circuit.append_gate(TdgGate(), [1])
    circuit.append_gate(CNOTGate(), [0, 1])
    return circuit


def circuit_gen(
    size: int,
    radixes: Sequence[int] = [],
    depth: int = 10,
    constant: bool = False,
    gateset: list[Gate] = BQSKIT_GATES,
) -> Circuit:
    """
    Generate a random circuit according to input specifications.

    Args:
        size (int): The number of qudits in the circuit.

        radixes (Sequence[int]): A sequence with its length equal
            to `size`. Each element specifies the base of a
            qudit. Defaults to qubits.

        depth (int): The number of gates in the circuit. (Default: 10)

        constant (bool): True implies the circuit will only consist of
            constant gates. (Default: False)

    Returns:
        (Circuit): The randomly generated circuit.
    """
    if not isinstance(depth, int):
        raise TypeError('Expected int for depth, got: %s.' % type(depth))

    if not isinstance(constant, bool):
        raise TypeError(
            'Expected bool for constant, got: %s.' %
            type(constant),
        )

    circuit = Circuit(size, radixes)

    # Apply a random gate to a random location `depth` times
    for d in range(depth):

        # 1. Select random qudit
        qudit_selected = np.random.randint(0, size)
        qudit_radix = circuit.radixes[qudit_selected]

        # 2. Select random gate and location
        gate_selected = None
        location_selected = None

        # 2a. Shuffle gates
        shuffled_gates = gateset
        np.random.shuffle(shuffled_gates)

        # 2b. Find first gate that is compatible
        for gate in shuffled_gates:

            # must be compatible with qudit
            if qudit_radix not in gate.radixes:
                continue

            # must be compatible with the constant flag
            if constant and gate.is_parameterized():
                continue

            # must be compatible with circuit size
            if gate.num_qudits > size:
                continue

            # must be compatible with circuit radix
            if not all(
                gate.radixes.count(unique_radix)
                <= circuit.radixes.count(unique_radix)
                for unique_radix in set(gate.radixes)
            ):
                continue

            gate_selected = gate
            break

        if gate_selected is None:  # IdentityGate(1) should always be valid
            raise RuntimeError('Should never be reached.')

        # 3. Select location for gate
        location_selected = []
        qudit_selected_index = gate.radixes.index(qudit_radix)

        for i, radix in enumerate(gate.radixes):
            # One qudit has already been matched
            if i == qudit_selected_index:
                location_selected.append(qudit_selected)
                continue

            # Find matching circuit qudit
            iter = 0
            qudit = None
            while (
                qudit is None
                or qudit in location_selected
                or qudit == qudit_selected
            ):
                qudit = circuit.radixes[iter:].index(radix) + iter
                iter = qudit + 1
            location_selected.append(qudit)

        # 4. Append gate
        params = np.random.random(gate_selected.num_params)
        circuit.append_gate(gate_selected, location_selected, params)

    return circuit


@pytest.fixture
def gen_random_circuit() -> Callable[[int, Sequence[int], int, bool], Circuit]:
    """Provide a function to generate random circuits."""
    return circuit_gen

# Pregenerated random circuit fixtures:


def random_3_qubit_circuits() -> list[tuple[int, Circuit]]:
    """Provide random 3-qubit circuits with 10 gates."""
    circuits = []
    for i in range(NUMBER_RANDOM_CIRCUITS):
        seed = int(3e8 + i)
        np.random.seed(seed)
        circuits.append((seed, circuit_gen(3)))
    return circuits


@pytest.fixture(params=random_3_qubit_circuits(), ids=lambda circ: circ[0])
def r3_qubit_circuit(request: Any) -> Circuit:
    """Provide random 3-qubit circuits as a fixture."""
    return request.param[1].copy()


def random_3_qubit_constant_circuits() -> list[tuple[int, Circuit]]:
    """Provide random 3-qubit constant circuits with 25 gates."""
    circuits = []
    for i in range(NUMBER_RANDOM_CIRCUITS):
        seed = int(7e8 + i)
        np.random.seed(seed)
        circuits.append((seed, circuit_gen(3, depth=25, constant=True)))
    return circuits


@pytest.fixture(
    params=random_3_qubit_constant_circuits(),
    ids=lambda circ: circ[0],
)
def r3_qubit_constant_circuit(request: Any) -> Circuit:
    """Provide random 3-qubit constant circuits as a fixture."""
    return request.param[1].copy()


def random_3_qutrit_circuits() -> list[tuple[int, Circuit]]:
    """Provide random 3-qutrit circuits with 10 gates."""
    circuits = []
    for i in range(NUMBER_RANDOM_CIRCUITS):
        seed = int(34e8 + i)
        np.random.seed(seed)
        circuits.append((seed, circuit_gen(3, [3] * 3)))
    return circuits


@pytest.fixture(params=random_3_qutrit_circuits(), ids=lambda circ: circ[0])
def r3_qutrit_circuit(request: Any) -> Circuit:
    """Provide random 3-qutrit circuits as a fixture."""
    return request.param[1].copy()


def random_6_qudit_circuits() -> list[tuple[int, Circuit]]:
    """Provide random 6-qudit random-radix circuits with 10 gates."""
    circuits = []
    for i in range(NUMBER_RANDOM_CIRCUITS):
        seed = int(14e8 + i)
        np.random.seed(seed)
        radixes = np.random.choice([2, 3], 6)
        circuits.append((seed, circuit_gen(6, radixes)))
    return circuits


@pytest.fixture(params=random_6_qudit_circuits(), ids=lambda circ: circ[0])
def r6_qudit_circuit(request: Any) -> Circuit:
    """Provide random 6-qudit random-radix circuits as a fixture."""
    return request.param[1].copy()


# endregion
