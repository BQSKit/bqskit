"""This module implements the SeedLayerGenerator class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.ir.circuit import Circuit
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.generators.simple import SimpleLayerGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
_logger = logging.getLogger(__name__)


class SeedLayerGenerator(LayerGenerator):
    """Layer Generator for search that starts from a seed."""

    def __init__(
        self,
        seed: Circuit,
        forward_generator: LayerGenerator = SimpleLayerGenerator(),
        num_removed: int = 1,
    ) -> None:
        """
        Construct a SeedLayerGenerator.

        Args:
            seed (Circuit): The seed to start from.

            forward_generator (Gate): A generator used to grow the circuit.

            num_removed (int): The number of gates removed from the circuit
                in each backwards branch.
        """
        if not isinstance(seed, Circuit):
            raise TypeError(f'Expected Circuit for seed, got {type(seed)}')

        if not isinstance(forward_generator, LayerGenerator):
            raise TypeError(
                'Expected LayerGenerator for forward_generator'
                f', got {type(forward_generator)}.',
            )

        if not is_integer(num_removed):
            raise TypeError(
                f'Expected integer for num_removed, got {type(num_removed)}.',
            )

        self.seed = seed
        self.forward_generator = forward_generator
        self.num_removed = num_removed

    def gen_initial_layer(
        self,
        target: UnitaryMatrix | StateVector,
        data: dict[str, Any],
    ) -> Circuit:
        """
        Generate the initial layer, see LayerGenerator for more.

        Raises:
            ValueError: If `target` has a size or radix mismatch with
                `self.seed`.
        """

        if not isinstance(target, (UnitaryMatrix, StateVector)):
            raise TypeError(
                'Expected unitary or state, got %s.' % type(target),
            )

        if target.dim != self.seed.dim:
            raise ValueError('Seed dimension mismatch with target.')

        data['seed_seen_before'] = {self.hash_structure(self.seed)}

        return self.seed

    def gen_successors(
        self,
        circuit: Circuit,
        data: dict[str, Any],
    ) -> list[Circuit]:
        """
        Generate the successors of a circuit node.

        Raises:
            ValueError: If circuit is a single-qudit circuit.
        """

        if not isinstance(circuit, Circuit):
            raise TypeError('Expected circuit, got %s.' % type(circuit))

        if circuit.num_qudits < 2:
            raise ValueError('Cannot expand a single-qudit circuit.')

        if circuit.dim != self.seed.dim:
            raise ValueError('Seed dimension mismatch with circuit.')

        # Generate successors
        successors = self.forward_generator.gen_successors(circuit, data)

        removed_count = 0
        for cycle, op in circuit.operations_with_cycles(reverse=True):
            removed_count += 1
            if removed_count > self.num_removed:
                break
            copied_circuit = circuit.copy()
            copied_circuit.pop((cycle, op.location[0]))
            successors.insert(0, copied_circuit)

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
