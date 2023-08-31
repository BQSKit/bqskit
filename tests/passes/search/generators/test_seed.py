from __future__ import annotations

from hypothesis import given

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import U3Gate
from bqskit.passes.search.generators import SeedLayerGenerator
from bqskit.utils.test.strategies import circuits


@given(circuits(2))
def test_seed_hash(circuit: Circuit) -> None:
    hash1 = SeedLayerGenerator.hash_structure(circuit)
    circuit.append_gate(HGate(), 0)
    hash2 = SeedLayerGenerator.hash_structure(circuit)
    assert hash1 != hash2


def test_constructor() -> None:
    seed_1 = Circuit(2)
    seed_2 = Circuit(2)
    seed_1.append_gate(CNOTGate(), (0, 1))
    seed_2.append_gate(CNOTGate(), (0, 1))
    seed_2.append_gate(CNOTGate(), (1, 0))
    no_seed = SeedLayerGenerator()
    one_seed = SeedLayerGenerator(seed_1)
    two_seed = SeedLayerGenerator([seed_1, seed_2])

    assert len(no_seed.seeds) == 0
    assert one_seed.seeds[0] == seed_1
    assert two_seed.seeds[0] == seed_1
    assert two_seed.seeds[1] == seed_2


def test_gen_initial_layer() -> None:
    layer_gen = SeedLayerGenerator()
    for num_q in range(2, 5):
        empty_circuit = Circuit(num_q)
        data = PassData(empty_circuit)
        init_circ = layer_gen.gen_initial_layer(
            empty_circuit.get_unitary(), data,
        )
        assert init_circ.num_operations == 0
        assert init_circ.num_qudits == num_q


def test_remove_atomic_units() -> None:
    num_q = 4
    circ = Circuit(num_q)
    for i in range(num_q):
        circ.append_gate(U3Gate(), [i])
    for edge in [(0, 1), (2, 3)]:
        circ.append_gate(CNOTGate(), edge)
    for i in range(num_q):
        circ.append_gate(U3Gate(), [i])

    layer_gen = SeedLayerGenerator(num_removed=2)

    one_left_circ, zero_left_circ = layer_gen.remove_atomic_units(circ)

    assert one_left_circ.num_operations == 7
    assert zero_left_circ.num_operations == 4
    assert isinstance(one_left_circ.get_operation((1, 0)).gate, CNOTGate)
    assert isinstance(one_left_circ.get_operation((2, 0)).gate, U3Gate)
    assert isinstance(one_left_circ.get_operation((2, 1)).gate, U3Gate)
    assert zero_left_circ.depth == 1


def test_gen_successors() -> None:
    # Set two seeds, assert that they are returned upon the first call to
    # gen_successors.
    seed_1 = Circuit(2)
    seed_2 = Circuit(2)
    seed_1.append_gate(CNOTGate(), (0, 1))
    seed_2.append_gate(CNOTGate(), (0, 1))
    seed_2.append_gate(CNOTGate(), (1, 0))

    seeds = [seed_1, seed_2]
    layer_gen = SeedLayerGenerator(seed=seeds)

    target = seed_2.get_unitary()
    data = PassData(seed_2)

    init_layer = layer_gen.gen_initial_layer(target, data)

    assert init_layer.num_operations == 0

    successors = layer_gen.gen_successors(init_layer, data)

    assert successors[0] == seed_1
    assert successors[1] == seed_2

    # Check that gen_successors works for non initial circuits
    successors = layer_gen.gen_successors(seed_2, data)
    # seed_1 should not be returned because it has already been seed
    assert len(successors) == 1

    assert successors[0].num_operations == 5
    assert isinstance(successors[0].get_operation((0, 0)).gate, CNOTGate)
    assert isinstance(successors[0].get_operation((1, 0)).gate, CNOTGate)
    assert isinstance(successors[0].get_operation((2, 0)).gate, CNOTGate)
    assert isinstance(successors[0].get_operation((3, 0)).gate, U3Gate)
    assert isinstance(successors[0].get_operation((3, 1)).gate, U3Gate)


def test_gen_successors_mismatch() -> None:
    # Have a 2 qubit and 3 qubit seed
    # Pass in a 3 qubit target and ensure only the 3 qubit seed is used
    seed_1 = Circuit(2)
    seed_2 = Circuit(3)
    seed_1.append_gate(CNOTGate(), (0, 1))
    seed_2.append_gate(CNOTGate(), (0, 1))
    seed_2.append_gate(CNOTGate(), (1, 2))

    seeds = [seed_1, seed_2]
    layer_gen = SeedLayerGenerator(seed=seeds)

    target = seed_2.get_unitary()
    data = PassData(seed_2)

    init_layer = layer_gen.gen_initial_layer(target, data)

    assert init_layer.num_operations == 0

    successors = layer_gen.gen_successors(init_layer, data)

    assert len(successors) == 1
    assert successors[0] == seed_2
