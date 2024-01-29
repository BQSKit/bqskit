"""This module implements the SeedLayerGenerator class."""
from __future__ import annotations

import logging
from typing import cast
from typing import Sequence

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.point import CircuitPoint
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.generators.simple import SimpleLayerGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_sequence

_logger = logging.getLogger(__name__)


class SeedLayerGenerator(LayerGenerator):
    """Layer Generator for search that starts from a seed."""

    def __init__(
        self,
        seed: Circuit | Sequence[Circuit],
        forward_generator: LayerGenerator = SimpleLayerGenerator(),
        num_removed: int = 1,
    ) -> None:
        """
        Construct a SeedLayerGenerator.

        Args:
            seed (Circuit | Sequence[Circuit]): The seed or seeds
                to start synthesis from.

            forward_generator (LayerGenerator): A generator used to grow
                the circuit. (Default: SimpleLayerGenerator)

            num_removed (int): The number of atomic gate units removed
                from the circuit in each backwards branch. If 0, no
                backwards traversal of the synthesis tree will be done.
                (Default: 1)

        Raises:
            ValueError: If 'num_removed' is negative.
        """
        if not isinstance(forward_generator, LayerGenerator):
            raise TypeError(
                'Expected LayerGenerator for forward_generator'
                f', got {type(forward_generator)}.',
            )

        if not is_integer(num_removed):
            raise TypeError(
                'Expected integer for num_removed, '
                f'got {type(num_removed)}.',
            )

        if num_removed < 0:
            raise ValueError(
                'Expected non-negative value for num_removed, '
                f'got {num_removed}.',
            )

        if isinstance(seed, Circuit):
            seed = [seed]

        if not is_sequence(seed):
            raise TypeError(
                'Expected a Circuit or Sequence of Circuits for seed, '
                f'got {type(seed)}.',
            )

        if not all(isinstance(s, Circuit) for s in seed):
            msg = 'Expected a Circuit or Sequence of Circuits for seed.'
            for i, s in enumerate(seed):
                if not isinstance(s, Circuit):
                    msg += f'\nGot Sequence with {type(s)} at index {i}.'
            raise TypeError(msg)

        self.seeds = list(cast(Sequence[Circuit], seed))
        self.forward_generator = forward_generator
        self.num_removed = num_removed

    def gen_initial_layer(
        self,
        target: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
    ) -> Circuit:
        """
        Generate the initial layer, see LayerGenerator for more.

        Raises:
            TypeError: If target is not a UnitaryMatrix, StateVector, or
                StateSystem.

        Notes:
            - For seeded layer generation, the initial_layer is the
              empty Circuit. The `gen_successors` method checks for this
              before returning `self.seeds` as successors.
        """
        if not isinstance(target, (UnitaryMatrix, StateVector, StateSystem)):
            raise TypeError(f'Expected unitary or state, got {type(target)}.')

        return Circuit(target.num_qudits, target.radixes)

    def gen_successors(self, circuit: Circuit, data: PassData) -> list[Circuit]:
        """
        Generate the successors of a circuit node.

        If `circuit` is the empty Circuit, seeds with radixes matching
        `circuit` will be used. If no seeds match, the empty Circuit will
        be returned as the only successor with a warning message.
        """
        if not isinstance(circuit, Circuit):
            raise TypeError(f'Expected Circuit, got {type(circuit)}.')

        if 'seed_seen_before' not in data:
            data['seed_seen_before'] = set()

        # If circuit is empty Circuit, successors are the seeds
        if circuit.is_empty:
            circ_hash = self.hash_structure(circuit)
            # Do not return seeds if empty circuit already visited
            if circ_hash in data['seed_seen_before']:
                return []
            data['seed_seen_before'] = {circ_hash}

            # Check if any seeds match circuit, only use those seeds
            usable_seeds = [
                seed for seed in self.seeds
                if circuit.radixes == seed.radixes
            ]
            for seed in usable_seeds:
                data['seed_seen_before'].add(self.hash_structure(seed))

            if len(usable_seeds) == 0:
                _logger.warning(
                    'No seeds matching the circuit\'s radixes found.'
                    '\nGenerating successors from empty circuit.'
                    '\nThis may cause a malformed search tree because the'
                    'generator never generated a proper initial layer.',
                )
                usable_seeds.append(circuit)

            return usable_seeds

        # Generate successors
        successors = self.forward_generator.gen_successors(circuit, data)

        # Search reverse direction
        ancestor_circuits = self.remove_atomic_units(circuit)
        successors = ancestor_circuits + successors

        # Filter out successors that have already been visited
        filtered_successors = []
        for s in successors:
            h = self.hash_structure(s)
            if h not in data['seed_seen_before']:
                data['seed_seen_before'].add(h)
                filtered_successors.append(s)

        return filtered_successors

    @staticmethod
    def hash_structure(circuit: Circuit) -> int:
        hashes = []
        for cycle, op in circuit.operations_with_cycles():
            if op.num_qudits <= 1:
                continue
            hashes.append(hash((cycle, str(op))))
            if len(hashes) > 100:
                hashes = [sum(hashes)]
        return sum(hashes)

    def remove_atomic_units(self, circuit: Circuit) -> list[Circuit]:
        """
        Return circuits after removing upto `self.num_removed` atomic units.

        Atomic units defined as multi-qudit gates and the single-qudit
        gates that are directly dependent on them. Except for single-qudit
        circuits, where the single-qudit gate is the atomic unit.

        For two qudit synthesis, these atomic units look like:

            -- two_qudit_gate -- single_qudit_gate_1 --
                    |
            -- two_qudit_gate -- single_qudit_gate_2 --

        Generally, this will find the last `num_removed` multi-qudit
        gates, and remove them and any single qudit gates that are
        directly dependent on them.
        """
        num_removed = 0
        ancestor_circuits = []

        circuit_copy = circuit.copy()

        if circuit_copy.num_qudits == 1:
            init_num_cycles = circuit_copy.num_cycles
            for _ in range(min(self.num_removed, init_num_cycles)):
                circuit_copy.pop_cycle(-1)
                ancestor_circuits.append(circuit_copy.copy())

        for cycle, op in circuit.operations_with_cycles(reverse=True):

            if num_removed >= self.num_removed:
                break
            if op.num_qudits == 1:
                continue

            # Remove multi-qudit gate and single qudit dependents
            point = CircuitPoint(cycle, op.location[0])
            dependents = []
            for next_point in circuit_copy.next(point):
                if circuit_copy.get_operation(next_point).num_qudits == 1:
                    dependents.append(next_point)
            to_remove = dependents + [point]

            circuit_copy.batch_pop(to_remove)

            ancestor_circuits.append(circuit_copy.copy())
            num_removed += 1

        return ancestor_circuits
