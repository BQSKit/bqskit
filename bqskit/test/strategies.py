"""This module implements hypothesis strategies for BQSKit."""
from __future__ import annotations

import inspect
from typing import Any
from typing import Sequence

from hypothesis.control import assume
from hypothesis.strategies import composite
from hypothesis.strategies import deferred
from hypothesis.strategies import dictionaries
from hypothesis.strategies import floats
from hypothesis.strategies import from_type
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import one_of
from hypothesis.strategies import sampled_from
from hypothesis.strategies import text
from hypothesis.strategies._internal.core import permutations
from hypothesis.strategies._internal.strategies import SearchStrategy

import bqskit.ir.gates
from bqskit.ir import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates.composed.daggergate import DaggerGate
from bqskit.ir.gates.composed.frozenparam import FrozenParameterGate
from bqskit.ir.gates.composed.tagged import TaggedGate
from bqskit.ir.gates.constant.identity import IdentityGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.gates.parameterized.pauli import PauliGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.interval import CycleInterval
from bqskit.ir.point import CircuitPoint
from bqskit.ir.region import CircuitRegion
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
def unitaries(
    draw: Any,
    size: int,
    radixes: Sequence[int] = [],
) -> UnitaryMatrix:
    """Hypothesis strategy for generating `UnitaryMatrix`'s."""
    return UnitaryMatrix.random(size, radixes)


@composite
def identity_gates(
    draw: Any,
    size: int,
    radixes: Sequence[int] = [],
) -> IdentityGate:
    """Hypothesis strategy for generating `IdentityGate`'s."""
    return IdentityGate(size, radixes)


@composite
def constant_unitary_gates(
    draw: Any,
    radixes: Sequence[int],
) -> ConstantUnitaryGate:
    """Hypothesis strategy for generating `ConstantUnitaryGate`'s."""
    utry = draw(unitaries(len(radixes), radixes))
    return ConstantUnitaryGate(utry, radixes)


@composite
def pauli_gates(draw: Any, size: int) -> PauliGate:
    """Hypothesis strategy for generating `PauliGate`'s."""
    return PauliGate(size)


@composite
def variable_unitary_gates(
    draw: Any,
    size: int,
    radixes: Sequence[int] = [],
) -> VariableUnitaryGate:
    """Hypothesis strategy for generating `VariableUnitaryGate`'s."""
    return VariableUnitaryGate(size, radixes)


# TODO: ControlledGate (gate, num_controls, radixes)


@composite
def dagger_gates(
    draw: Any,
    radixes: Sequence[int] | int | None = None,
    constant: bool | None = None,
) -> DaggerGate:
    """Hypothesis strategy for generating `DaggerGate`'s."""
    gate = draw(deferred(lambda: gates(radixes, constant)))
    return DaggerGate(gate)


@composite
def frozen_gates(
    draw: Any,
    radixes: Sequence[int] | int | None = None,
) -> FrozenParameterGate:
    """Hypothesis strategy for generating `FrozenParameterGate`'s."""
    gate = draw(deferred(lambda: gates(radixes, False)))
    max_idx = gate.num_params
    indices = integers(0, max_idx - 1)
    values = floats(allow_nan=False, allow_infinity=False, width=16)
    frozen_params = draw(dictionaries(indices, values, max_size=max_idx))
    return FrozenParameterGate(gate, frozen_params)


@composite
def tagged_gates(
    draw: Any,
    radixes: Sequence[int] | int | None = None,
    constant: bool | None = None,
) -> TaggedGate:
    """Hypothesis strategy for generating `TaggedGate`'s."""
    gate = draw(deferred(lambda: gates(radixes, constant)))
    tag = draw(text())
    return TaggedGate(gate, tag)


# TODO: CircuitGate (circuit, move)


@composite
def gates(
    draw: Any,
    radixes: Sequence[int] | int | None = None,
    constant: bool | None = None,
) -> Gate:
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

    assume(sorted(gate.radixes) == sorted(radixes))
    assume(gate.num_params <= 128)

    return gate


@composite
def circuits(
    draw: Any,
    radixes: Sequence[int] | int | None = None,
    max_gates: int = 10,
    exact_num_gates: bool = False,
    constant: bool = False,
) -> Circuit:
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
        params = floats(allow_nan=False, allow_infinity=False, width=16)
        num_params = gate.num_params
        gate_params = draw(
            lists(
                params,
                min_size=num_params,
                max_size=num_params,
            ),
        )
        circuit.append_gate(gate, gate_location, gate_params)

    return circuit


@composite
def circuit_points(
    draw: Any,
    max_cycle: int = 1000,
    max_qudit: int = 1000,
) -> CircuitPoint:
    """Hypothesis strategy for generating `CircuitPoint`'s."""
    cycle = draw(integers(0, max_cycle))
    qudit = draw(integers(0, max_qudit))
    return CircuitPoint(cycle, qudit)


@composite
def cycle_intervals(
    draw: Any,
    max_max_cycle: int = 1000,
    max_range: int = 100,
) -> CycleInterval:
    """Hypothesis strategy for generating `CycleInterval`'s."""
    lower = draw(integers(0, max_max_cycle))
    upper = draw(
        integers(lower, min(max_max_cycle, lower + max_range)),
    )
    return CycleInterval(lower, upper)


@composite
def circuit_regions(
    draw: Any,
    max_max_cycle: int = 1000,
    max_qudits: int = 20,
    max_volume: int = 200,
    empty: bool | None = None,
) -> CircuitRegion:
    """Hypothesis strategy for generating `CircuitRegion`'s."""
    if empty is True:
        return CircuitRegion({})

    region = draw(
        dictionaries(
            integers(0),
            cycle_intervals(max_max_cycle, max_volume // max_qudits),
            min_size=0 if empty is None else 1,
            max_size=max_qudits,
        ),
    )
    return CircuitRegion(region)


def everything_except(
    excluded_types: type | tuple[type, ...],
) -> SearchStrategy[Any]:
    """
    Hypothesis strategy to generate inputs of types not in `excluded_types`.

    References:
        https://hypothesis.readthedocs.io/en/latest/data.html
    """
    if not isinstance(excluded_types, tuple):
        excluded_types = tuple([excluded_types])

    checked_types = []
    for excluded_type in excluded_types:
        try:
            isinstance(0, excluded_type)
            checked_types.append(excluded_type)
        except TypeError:
            continue

    if len(checked_types) == 0:
        return from_type(type).flatmap(from_type)

    return (
        from_type(type)
        .flatmap(from_type)
        .filter(
            lambda x: not isinstance(x, tuple(checked_types)),  # type: ignore
        )
    )
