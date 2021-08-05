# type: ignore
"""This module implements hypothesis strategies for BQSKit."""
from __future__ import annotations

import inspect
from typing import Sequence

from hypothesis.control import assume
from hypothesis.strategies import composite
from hypothesis.strategies import deferred
from hypothesis.strategies import dictionaries
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import one_of
from hypothesis.strategies import sampled_from
from hypothesis.strategies import text
from hypothesis.strategies._internal.core import permutations

import bqskit.ir.gates
from bqskit.ir import Circuit
from bqskit.ir.gates.composed.daggergate import DaggerGate
from bqskit.ir.gates.composed.frozenparam import FrozenParameterGate
from bqskit.ir.gates.composed.tagged import TaggedGate
from bqskit.ir.gates.constant.identity import IdentityGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.gates.parameterized.pauli import PauliGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_sequence_of_int

gate_instances = []
for gate_class_str in bqskit.ir.gates.__all__:
    gate_class = getattr(bqskit.ir.gates, gate_class_str)
    gate_params = inspect.signature(gate_class.__init__).parameters
    if (
        len(gate_params) == 3
        and 'self' in gate_params
        and 'args' in gate_params
        and 'kwargs' in gate_params
    ):
        gate_instances.append(gate_class())


simple_gates = sampled_from(gate_instances)


@composite
def unitaries(draw, size: int, radixes: Sequence[int] = []):
    """Hypothesis strategy for generating `UnitaryMatrix`s."""
    return UnitaryMatrix.random(size, radixes)


@composite
def identity_gates(draw, size: int, radixes: Sequence[int] = []):
    """Hypothesis strategy for generating `IdentityGate`s."""
    return IdentityGate(size, radixes)


@composite
def constant_unitary_gates(draw, radixes: Sequence[int]):
    """Hypothesis strategy for generating `ConstantUnitaryGate`s."""
    utry = draw(unitaries(len(radixes), radixes))
    return ConstantUnitaryGate(utry, radixes)


@composite
def pauli_gates(draw, size: int):
    """Hypothesis strategy for generating `PauliGate`s."""
    return PauliGate(size)


@composite
def variable_unitary_gates(draw, size: int, radixes: Sequence[int] = []):
    """Hypothesis strategy for generating `VariableUnitaryGate`s."""
    return VariableUnitaryGate(size, radixes)


# TODO: ControlledGate (gate, num_controls, radixes)


@composite
def dagger_gates(
    draw,
    radixes: Sequence[int] | int | None = None,
    constant: bool | None = None,
):
    """Hypothesis strategy for generating `DaggerGate`s."""
    gate = draw(deferred(lambda: gates(radixes, constant)))
    return DaggerGate(gate)


@composite
def frozen_gates(
    draw,
    radixes: Sequence[int] | int | None = None,
):
    """Hypothesis strategy for generating `FrozenParameterGate`s."""
    gate = draw(deferred(lambda: gates(radixes, False)))
    max_idx = gate.get_num_params()
    indices = integers(0, max_idx - 1)
    values = floats(allow_nan=False, allow_infinity=False)
    frozen_params = draw(dictionaries(indices, values, max_size=max_idx))
    return FrozenParameterGate(gate, frozen_params)


@composite
def tagged_gates(
    draw,
    radixes: Sequence[int] | int | None = None,
    constant: bool | None = None,
):
    """Hypothesis strategy for generating `TaggedGate`s."""
    gate = draw(deferred(lambda: gates(radixes, constant)))
    tag = draw(text())
    return TaggedGate(gate, tag)


# TODO: CircuitGate (circuit, move)


@composite
def gates(
    draw,
    radixes: Sequence[int] | int | None = None,
    constant: bool | None = None,
):
    """Hypothesis strategy for generating gates."""
    if radixes is None:
        radixes = 2

    if is_integer(radixes):
        radixes = [radixes] * draw(sampled_from([1, 2, 3, 4]))

    if not is_sequence_of_int(radixes):
        raise TypeError(
            'Expected sequence of integers or integer for radixes,'
            f'got {type(radixes)}.',
        )

    size = len(radixes)

    gate = draw(
        one_of(
            simple_gates,
            identity_gates(size, radixes),
            constant_unitary_gates(radixes),
            pauli_gates(size),
            variable_unitary_gates(size, radixes),
            dagger_gates(radixes, constant),
            tagged_gates(radixes, constant),
            frozen_gates(radixes),
        ),
    )

    if constant is not None:
        assume(gate.is_constant() == constant)

    assume(sorted(gate.get_radixes()) == sorted(radixes))
    assume(gate.get_num_params() <= 128)

    return gate


@composite
def circuits(
    draw,
    radixes: Sequence[int] | int | None = None,
    max_gates: int = 10,
    exact_num_gates: bool = False,
    constant: bool = False,
):
    """Hypothesis strategy for generating circuits."""

    if not isinstance(max_gates, int):
        raise TypeError(f'Expected int for max_gates, got: f{type(max_gates)}.')

    if not isinstance(constant, bool):
        raise TypeError(f'Expected bool for constant, got: {type(constant)}.')

    if radixes is None:
        radixes = 2

    if is_integer(radixes):
        radixes = [radixes] * draw(sampled_from([1, 2, 3, 4, 5, 6]))

    if not is_sequence_of_int(radixes):
        raise TypeError(
            'Expected sequence of integers or integer for radixes,'
            f'got {type(radixes)}.',
        )

    if exact_num_gates:
        num_gates = max_gates
    else:
        num_gates = draw(sampled_from(list(range(max_gates))))
    size = len(radixes)
    circuit = Circuit(size, radixes)

    for d in range(num_gates):
        gate_size = draw(sampled_from([1, 2, 3, 4]))
        qubit_indices = list(range(size))
        idx_and_rdx = list(zip(qubit_indices, radixes))
        gate_idx_and_rdx = draw(permutations(idx_and_rdx))[:gate_size]
        gate_location = list(zip(*gate_idx_and_rdx))[0]
        gate_radixes = list(zip(*gate_idx_and_rdx))[1]
        gate = draw(gates(gate_radixes, constant))
        params = floats(allow_nan=False, allow_infinity=False)
        num_params = gate.get_num_params()
        gate_params = draw(
            lists(
                params,
                min_size=num_params,
                max_size=num_params,
            ),
        )
        circuit.append_gate(gate, gate_location, gate_params)

    return circuit
