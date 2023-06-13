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
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import one_of
from hypothesis.strategies import permutations
from hypothesis.strategies import sampled_from
from hypothesis.strategies import SearchStrategy
from hypothesis.strategies import text
from hypothesis.strategies import tuples

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
from bqskit.ir.location import CircuitLocation
from bqskit.ir.location import CircuitLocationLike
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.ir.point import CircuitPointLike
from bqskit.ir.region import CircuitRegion
from bqskit.qis.state.state import StateLike
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.qis.unitary.unitarymatrix import UnitaryLike
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_sequence_of_int


@composite
def num_qudits(
    draw: Any,
    max_num_qudits: int = 8,
    min_num_qudits: int = 1,  # TODO: Param order should be swapped
) -> int:
    """Hypothesis strategy for generating a valid number of qudits."""
    return draw(integers(min_num_qudits, max_num_qudits))


@composite
def radixes(
    draw: Any,
    max_num_qudits: int = 4,
    allowed_bases: Sequence[int] = (2, 3, 4),
    min_num_qudits: int = 1,
) -> tuple[int, ...]:
    """Hypothesis strategy for generating a valid radixes object."""
    num_q = draw(num_qudits(max_num_qudits, min_num_qudits))
    x = [sampled_from(allowed_bases) for _ in range(num_q)]
    return draw(tuples(*x))


@composite
def num_qudits_and_radixes(
    draw: Any,
    max_num_qudits: int = 4,
    allowed_bases: Sequence[int] = (2, 3, 4),
    min_num_qudits: int = 1,
) -> tuple[int, tuple[int, ...]]:
    """Hypothesis strategy for a matching pair of num_qudits and radixes."""
    num_q = draw(num_qudits(max_num_qudits, min_num_qudits))
    x = [sampled_from(allowed_bases) for _ in range(num_q)]
    return (num_q, draw(tuples(*x)))


@composite
def unitaries(
    draw: Any,
    max_num_qudits: int = 3,
    allowed_bases: Sequence[int] = (2, 3),
    min_num_qudits: int = 1,
) -> UnitaryMatrix:
    """Hypothesis strategy for generating `UnitaryMatrix`'s."""
    num_qudits, radixes = draw(
        num_qudits_and_radixes(
            max_num_qudits, allowed_bases, min_num_qudits,
        ),
    )
    return UnitaryMatrix.random(num_qudits, radixes)


@composite
def unitary_likes(
    draw: Any,
    max_num_qudits: int = 3,
    allowed_bases: Sequence[int] = (2, 3),
    min_num_qudits: int = 1,
) -> UnitaryLike:
    """Hypothesis strategy for generating UnitaryLike objects."""
    utry = draw(unitaries(max_num_qudits, allowed_bases, min_num_qudits))
    return draw(sampled_from([utry, utry.numpy]))


@composite
def state_vectors(
    draw: Any,
    max_num_qudits: int = 3,
    allowed_bases: Sequence[int] = (2, 3),
    min_num_qudits: int = 1,
) -> StateVector:
    """Hypothesis strategy for generating `StateVector`'s."""
    num_qudits, radixes = draw(
        num_qudits_and_radixes(
            max_num_qudits, allowed_bases, min_num_qudits,
        ),
    )
    return StateVector.random(num_qudits, radixes)


@composite
def state_likes(
    draw: Any,
    max_num_qudits: int = 3,
    allowed_bases: Sequence[int] = (2, 3),
    min_num_qudits: int = 1,
) -> StateLike:
    """Hypothesis strategy for generating StateLike objects."""
    vec = draw(state_vectors(max_num_qudits, allowed_bases, min_num_qudits))
    return draw(sampled_from([vec, vec.numpy]))


gate_instances: list[Gate] = []
for gate_class_str in bqskit.ir.gates.__all__:
    gate_class = getattr(bqskit.ir.gates, gate_class_str)
    gate_params = inspect.signature(gate_class.__init__).parameters
    if (
        len(gate_params) == 3
        and 'self' in gate_params
        and 'args' in gate_params
        and 'kwargs' in gate_params
        and not inspect.isabstract(gate_class)
        and not gate_class.__name__ == 'ConstantGate'
    ):
        gate_instances.append(gate_class())


@composite
def constant_unitary_gates(
    draw: Any,
    radixes: Sequence[int],
) -> ConstantUnitaryGate:
    """Hypothesis strategy for generating `ConstantUnitaryGate`'s."""
    utry = draw(unitaries(len(radixes), radixes, len(radixes)))
    return ConstantUnitaryGate(utry, radixes)


@composite
def dagger_gates(
    draw: Any,
    radixes: Sequence[int] | int | None = None,
    constant: bool = False,
) -> DaggerGate:
    """Hypothesis strategy for generating `DaggerGate`'s."""
    gate = draw(deferred(lambda: gates(radixes, constant)))
    return DaggerGate(gate)


@composite
def frozen_gates(
    draw: Any,
    radixes: Sequence[int] | int | None = None,
    constant: bool = False,
) -> FrozenParameterGate:
    """Hypothesis strategy for generating `FrozenParameterGate`'s."""
    gate = draw(deferred(lambda: gates(radixes, False)))
    if gate.num_params == 0:
        return FrozenParameterGate(gate, {})
    max_idx = gate.num_params
    indices = integers(0, max_idx - 1)
    values = floats(allow_nan=False, allow_infinity=False, width=16)
    min_size = gate.num_params if constant else 0
    frozen_params = draw(
        dictionaries(
            indices,
            values,
            min_size=min_size,
            max_size=max_idx,
        ),
    )
    return FrozenParameterGate(gate, frozen_params)


@composite
def tagged_gates(
    draw: Any,
    radixes: Sequence[int] | int | None = None,
    constant: bool = False,
) -> TaggedGate:
    """Hypothesis strategy for generating `TaggedGate`'s."""
    gate = draw(deferred(lambda: gates(radixes, constant)))
    tag = draw(text())
    return TaggedGate(gate, tag)


# TODO: CircuitGate (circuit, move)
# TODO: ControlledGate (gate, num_controls, radixes)


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

    num_qudits = len(radixes)

    gate = draw(
        one_of(
            just(IdentityGate(num_qudits, radixes)),
            constant_unitary_gates(radixes),
            just(PauliGate(num_qudits)),
            just(VariableUnitaryGate(num_qudits, radixes)),
            sampled_from(gate_instances),
            dagger_gates(radixes, constant),
            tagged_gates(radixes, constant),
            frozen_gates(radixes, constant),
        ),
    )

    if constant is not None:
        assume(gate.is_constant() == constant)

    assume(sorted(gate.radixes) == sorted(radixes))
    assume(gate.num_params <= 128)

    return gate


@composite
def operations(
    draw: Any,
    radixes: Sequence[int] | int | None = None,
    constant: bool = False,
    max_qudit: int = 100,
) -> Operation:
    """Hypothesis strategy for generating operations."""
    gate = draw(gates(radixes, constant))
    location = draw(
        circuit_locations(
            max_qudit=max_qudit,
            min_size=gate.num_qudits,
            max_size=gate.num_qudits,
        ),
    )
    params = draw(
        one_of([
            lists(floats(), max_size=0),
            lists(
                floats(allow_nan=False, allow_infinity=False, width=16),
                min_size=gate.num_params,
                max_size=gate.num_params,
            ),
        ]),
    )
    return Operation(gate, location, params)


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
        radixes = [radixes] * draw(sampled_from([1, 2, 3, 4]))

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
    qubit_indices = list(range(size))
    idx_and_rdx = list(zip(qubit_indices, radixes))
    circuit = Circuit(size, radixes)

    for d in range(num_gates):
        gate_size = draw(sampled_from([1, 2, 3, 4]))
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
def circuit_locations(
    draw: Any,
    min_qudit: int = 0,
    max_qudit: int = 100,
    min_size: int = 1,
    max_size: int = 5,
) -> CircuitLocation:
    """Hypothesis strategy for generating `CircuitLocation` s."""
    idxs = integers(min_qudit, max_qudit)
    locations = lists(idxs, min_size=min_size, max_size=max_size, unique=True)
    return CircuitLocation(draw(locations))


@composite
def circuit_location_likes(
    draw: Any,
    min_qudit: int = 0,
    max_qudit: int = 100,
    min_size: int = 1,
    max_size: int = 5,
) -> CircuitLocationLike:
    """Hypothesis strategy for generating `CircuitLocationLike` s."""
    idxs = integers(min_qudit, max_qudit)
    locations = lists(idxs, min_size=min_size, max_size=max_size, unique=True)
    location = draw(locations)
    return draw(
        sampled_from(
            [location[0], location, CircuitLocation(location)],
        ),
    )


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
def circuit_point_likes(
    draw: Any,
    max_cycle: int = 1000,
    max_qudit: int = 1000,
) -> CircuitPointLike:
    """Hypothesis strategy for generating `CircuitPointLike` objects."""
    point = draw(circuit_points(max_cycle, max_qudit))
    return draw(sampled_from([point, tuple(point)]))


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
    max_num_qudits: int = 20,
    max_volume: int = 200,
    max_qudit: int = 1000,
    empty: bool | None = None,
) -> CircuitRegion:
    """Hypothesis strategy for generating `CircuitRegion`'s."""
    if empty is True:
        return CircuitRegion({})

    region = draw(
        dictionaries(
            integers(0, max_qudit),
            cycle_intervals(max_max_cycle, max_volume // max_num_qudits),
            min_size=0 if empty is None else 1,
            max_size=max_num_qudits,
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
        .filter(lambda x: not isinstance(x, tuple(checked_types)))
    )
