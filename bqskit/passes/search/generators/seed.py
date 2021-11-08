"""This module implements the SeedLayerGenerator class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.ir.circuit import Circuit
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.generators.simple import SimpleLayerGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
_logger = logging.getLogger(__name__)


class SeedLayerGenerator(LayerGenerator):
    """The SeedLayerGenerator class."""

    def __init__(
        self,
        seed: Circuit,
        forward_generator: LayerGenerator = SimpleLayerGenerator(),
    ) -> None:
        """
        Construct a SimpleLayerGenerator.

        Args:
            seed (Circuit): The seed to start from.

            forward_generator (Gate): A generator used to grow the circuit.
        """
        if not isinstance(seed, Circuit):
            raise TypeError(f'Expected Circuit for seed, got {type(seed)}')

        if not isinstance(forward_generator, LayerGenerator):
            raise TypeError(
                'Expected LayerGenerator for forward_generator'
                f', got {type(forward_generator)}.',
            )

        self.seed = seed
        self.forward_generator = forward_generator

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

        for cycle, op in circuit.operations_with_cycles():
            copied_circuit = circuit.copy()
            copied_circuit.pop((cycle, op.location[0]))
            successors.append(copied_circuit)

        return successors
