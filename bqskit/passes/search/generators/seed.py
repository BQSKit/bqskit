"""This module implements the SeedLayerGenerator class."""
from __future__ import annotations

import logging
from typing import Any
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

_logger = logging.getLogger(__name__)


class SeedLayerGenerator(LayerGenerator):
    """Layer Generator for search that starts from a seed."""

    def __init__(
        self,
        seed: Circuit | Sequence[Circuit] | None = None,
        forward_generator: LayerGenerator = SimpleLayerGenerator(),
        num_removed: int = 1,
    ) -> None:
        """
        Construct a SeedLayerGenerator.

        Args:
            seed (Circuit | Sequence[Circuit] | None): The seed or seeds
                to start synthesis from. (Default: None)

            forward_generator (LayerGenerator): A generator used to grow
                the circuit. (Default: SimpleLayerGenerator)

            num_removed (int): The number of atomic gate units removed
                from the circuit in each backwards branch. (Default: 1)

        Raises:
            TypeError: If `seed` are not of type None, Circuit,
                or Sequence[Circuit].

            TypeError: If `forward_generator` is not a LayerGenerator.

            TypeError: If `num_removed` is not an integer.

            ValueError: If 'num_removed' is negative.
        """
        self._check_seed_type(seed)

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

        if seed is None:
            self.seeds = []
        else:
            self.seeds = [seed] if not isinstance(
                seed, Sequence,
            ) else list(seed)
        self.forward_generator = forward_generator
        self.num_removed = num_removed

    def gen_initial_layer(
        self,
        target: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
    ) -> Circuit:
        """
        Generate the initial layer, see LayerGenerator for more.

        Note: For seeded layer generation, the initial_layer is the
            empty Circuit. The `gen_successors` method checks for this
            before returning `self.seeds` as successors.

        Raises:
            TypeError: If target is not a UnitaryMatrix, StateVector, or
                StateSystem.
        """
        if not isinstance(target, (UnitaryMatrix, StateVector, StateSystem)):
            raise TypeError(f'Expected unitary or state, got {type(target)}.')

        return Circuit(target.num_qudits)

    def gen_successors(
        self,
        circuit: Circuit,
        data: PassData,
    ) -> list[Circuit]:
        """
        Generate the successors of a circuit node. If `circuit` is the empty
        Circuit, seeds of the correct dimension will be used.

        Raises:
            TypeError: If `circuit` is not a Circuit.

            ValueError: If `circuit` is a single-qudit circuit.
        """
        if not isinstance(circuit, Circuit):
            raise TypeError(f'Expected Circuit, got {type(circuit)}.')

        # If circuit is empty Circuit, successors are seeds
        if self._is_empty_circuit(circuit):
            # Check if any seed dim matches, only use those seeds
            data['seed_seen_before'] = set()
            useable_seeds = []
            for seed in self.seeds:
                if circuit.dim != seed.dim:
                    continue
                useable_seeds.append(seed)
            # If there are no usable seeds, proceed to normal search
            if len(useable_seeds) > 0:
                for seed in useable_seeds:
                    data['seed_seen_before'].add(self.hash_structure(seed))
                return useable_seeds

        # Generate successors
        successors = self.forward_generator.gen_successors(circuit, data)

        # Search reverse direction
        ancestor_circuits = self.remove_atomic_units(circuit)
        successors = ancestor_circuits + successors

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
            hashes.append(hash((cycle, str(op))))
            if len(hashes) > 100:
                hashes = [sum(hashes)]
        return sum(hashes)

    def remove_atomic_units(self, circuit: Circuit) -> list[Circuit]:
        """
        Return circuits that correspond to removing upto `num_removed` synthesis
        atomic units.

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
        for cycle, op in circuit.operations_with_cycles(reverse=True):

            if num_removed >= self.num_removed:
                break
            if op.num_qudits == 1:
                continue

            # Remove multi-qudit gate and single qubit dependents
            point = CircuitPoint(cycle, op.location[0])
            dependents = []
            for next_point in circuit_copy.next(point):
                if circuit_copy.get_operation(next_point).num_qudits == 1:
                    dependents.append(next_point)
            to_remove = dependents + [point]

            for point in to_remove:
                if not circuit_copy.is_point_idle(point):
                    circuit_copy.pop(point)

            ancestor_circuits.append(circuit_copy)
            circuit_copy = circuit_copy.copy()
            num_removed += 1

        return ancestor_circuits

    def _check_seed_type(self, seed: Any) -> None:
        """
        Check that the seed provided is valid.

        Args:
            seed (Any): Seed element to check.

        Raises:
            ValueError: If seed is not None, a Circuit, or a Sequence of
                Circuits.
        """
        none_type = seed is None
        circ_type = isinstance(seed, Circuit)
        seq_type = isinstance(seed, Sequence)
        seq_circ_type = False
        if seq_type:
            seq_circ_type = all([isinstance(c, Circuit) for c in seed])

        if not none_type and not circ_type and not seq_circ_type:
            error_msg = (
                'Provided `seed` is not of a valid type (None, Circuit,'
                'Sequence[Circuit]). '
            )
            if seq_type and not seq_circ_type:
                error_msg += (
                    f'Provided a Sequence, but of type {type(seed[0])}.'
                )

            raise TypeError(error_msg)

    def _is_empty_circuit(self, circuit: Circuit) -> bool:
        return circuit.num_operations == 0
