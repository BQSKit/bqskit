"""This module implements the MiddleOutLayerGenerator class."""
from __future__ import annotations

import itertools as it
import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.passes.search.generators.seed import SeedLayerGenerator
from bqskit.passes.search.generators.simple import SimpleLayerGenerator

_logger = logging.getLogger(__name__)


# TODO: choose a better name?
class MiddleOutLayerGenerator(SimpleLayerGenerator):
    """Layer Generator for search that adds gates at each time step."""
    generated_circuits: set[int]

    def __init__(self) -> None:
        """Construct a MiddleOutLayerGenerator."""
        self.generated_circuits = set()
        super().__init__()

    def gen_successors(self, circuit: Circuit, data: PassData) -> list[Circuit]:
        """
        Generate the successors of a circuit node.

        Raises:
            ValueError: If circuit is a single-qudit circuit.
        """

        if not isinstance(circuit, Circuit):
            raise TypeError('Expected circuit, got %s.' % type(circuit))

        if circuit.num_qudits < 2:
            raise ValueError('Cannot expand a single-qudit circuit.')

        # Get the machine model
        coupling_graph = BasePass.get_connectivity(circuit, data)

        # Generate successors
        successors = []
        for (edge, cycle) in it.product(
                coupling_graph, range(circuit.num_cycles),
        ):
            successor = circuit.copy()
            successor.insert_gate(
                cycle, self.two_qudit_gate, [
                    edge[0], edge[1],
                ],
            )
            successor.insert_gate(cycle, self.single_qudit_gate_1, edge[0])
            successor.insert_gate(cycle, self.single_qudit_gate_2, edge[1])
            # TODO: evaluate whether or not this hash method is efficient
            hash = SeedLayerGenerator.hash_structure(successor)
            if hash not in self.generated_circuits:
                self.generated_circuits.add(hash)
                successors.append(successor)

        for edge in coupling_graph:
            successor = circuit.copy()
            successor.append_gate(self.two_qudit_gate, [edge[0], edge[1]])
            successor.append_gate(self.single_qudit_gate_1, edge[0])
            successor.append_gate(self.single_qudit_gate_2, edge[1])
            if hash not in self.generated_circuits:
                self.generated_circuits.add(hash)
                successors.append(successor)

        return successors
