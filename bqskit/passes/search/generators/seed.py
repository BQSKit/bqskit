"""This module implements the SeedLayerGenerator class."""
from __future__ import annotations

import logging
from typing import Sequence

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
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
        seeds: Circuit | Sequence[Circuit],
        forward_generator: LayerGenerator = SimpleLayerGenerator(),
        back_step_size: int = 1,
    ) -> None:
        """
        Construct a SeedLayerGenerator.

        Args:
            seeds (Circuit | Sequence[Circuit]): The seed or seeds to start
                synthesis from.

            forward_generator (LayerGenerator): A generator used to grow
                the circuit.

            back_step_size (int): The number of atomic gate units removed from
                the circuit in each backwards branch.

        Raises:
            TypeError: If `seeds` are not a Sequence of Circuits or a
                single Circuit.

            ValueError: If Circuits in `seeds` do not all have the same
                dimension.

            TypeError: If `forward_generator` is not a LayerGenerator.

            TypeError: If `back_step_size` is not an integer.

            ValueError: If 'back_step_size' is negative.
        """
        if not isinstance(seeds, Circuit) and not isinstance(seeds, Sequence):
            raise TypeError(
                f'Expected Circuit or Sequence of Circuits for '
                f'seed, got {type(seeds)} instead.',
            )

        if isinstance(seeds, Sequence):
            if not all([isinstance(c, Circuit) for c in seeds]):
                raise TypeError('Expected seed to be Sequence of Circuits.')
            self.seed_dim = seeds[0].dim
            if not all([s.dim == self.seed_dim for s in seeds]):
                raise ValueError('Each seed must be the same dimension.')
        else:
            self.seed_dim = seeds.dim

        if not isinstance(forward_generator, LayerGenerator):
            raise TypeError(
                'Expected LayerGenerator for forward_generator'
                f', got {type(forward_generator)}.',
            )

        if not is_integer(back_step_size):
            raise TypeError(
                'Expected integer for back_step_size, '
                f'got {type(back_step_size)}.',
            )
        if back_step_size < 0:
            raise ValueError(
                'Expected non-negative value for back_step_size, '
                f'got {back_step_size}.',
            )

        self.seeds = [seeds] if not isinstance(seeds, Sequence) else list(seeds)
        self.forward_generator = forward_generator
        self.back_step_size = back_step_size

    def gen_initial_layer(
        self,
        target: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
    ) -> list[Circuit]:
        """
        Generate the initial layer, see LayerGenerator for more.

        Raises:
            ValueError: If `target` has a size or radix mismatch with
                `self.seed`.
        """

        if not isinstance(target, (UnitaryMatrix, StateVector)):
            raise TypeError(f'Expected unitary or state, got {type(target)}.')

        if target.dim != self.seed_dim:
            raise ValueError('Seed dimension mismatch with target.')

        for seed in self.seeds:
            data['seed_seen_before'] = {self.hash_structure(seed)}

        return self.seeds

    def gen_successors(
        self,
        circuit: Circuit,
        data: PassData,
    ) -> list[Circuit]:
        """
        Generate the successors of a circuit node.

        Raises:
            ValueError: If circuit is a single-qudit circuit.
        """
        if not isinstance(circuit, Circuit):
            raise TypeError(f'Expected Circuit , got {type(circuit)}.')

        if circuit.dim != self.seed_dim:
            raise ValueError('Seed and circuit dimensions do not match.')

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
        Search for the last `back_step_size` number of atmoic units:

            -- two_qudit_gate -- single_qudit_gate_1 --
                    |
            -- two_qudit_gate -- single_qudit_gate_2 --

        and remove them.
        """
        num_removed = 0
        ancestor_circuits = []

        circuit_copy = circuit.copy()
        for cycle, op in circuit.operations_with_cycles(reverse=True):

            if num_removed >= self.back_step_size:
                break
            if op.num_qudits == 1:
                continue

            for place in op.location:
                point = (cycle + 1, place)
                if not circuit_copy.is_point_idle(point):
                    circuit_copy.pop(point)

            circuit_copy.pop((cycle, op.location[0]))

            ancestor_circuits.append(circuit_copy)
            circuit_copy = circuit_copy.copy()
            num_removed += 1

        return ancestor_circuits
