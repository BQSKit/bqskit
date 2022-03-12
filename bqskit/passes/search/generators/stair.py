"""This module implements the StairLayerGenerator class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.passes.search.generator import LayerGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


_logger = logging.getLogger(__name__)


class StairLayerGenerator(LayerGenerator):
    """Layer Generator for search that builds circuits from a single gate."""

    def __init__(self, gate: Gate) -> None:
        """
        Construct a StairLayerGenerator.

        Args:
            gate (Gate): The gate to build from.
        """
        if not isinstance(gate, Gate):
            raise TypeError(f'Expected Gate for gate, got {type(gate)}')

        self.gate = gate

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

        init_circuit = Circuit(target.num_qudits, target.radixes)

        # Place RXX Gates on consective pairs of qudits
        nq = self.gate.num_qudits
        for i in range(init_circuit.num_qudits - (nq - 1)):
            init_circuit.append_gate(self.gate, [i + j for j in range(nq)])

        return init_circuit

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

        nq = self.gate.num_qudits
        successor = circuit.copy()
        for i in range(circuit.num_qudits - (nq - 1)):
            successor.append_gate(self.gate, [i + j for j in range(nq)])

        return [successor]
