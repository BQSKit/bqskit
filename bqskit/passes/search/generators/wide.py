"""This module implements the WideLayerGenerator class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates import IToffoliGate
from bqskit.ir.gates import U3Gate
from bqskit.passes.search.generator import LayerGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
_logger = logging.getLogger(__name__)


class WideLayerGenerator(LayerGenerator):
    """
    A layer generator for use with wide gates like an iToffoli.

    Very similar to the simple layer generator. Each additional block adds one
    multi-qudit gate followed by single-qudit gates on each qudit.
    """

    def __init__(
        self,
        multi_qudit_gates: Gate | Sequence[Gate] = IToffoliGate(),
        single_qudit_gate: Gate = U3Gate(),
    ) -> None:
        """
        Construct a WideLayerGenerator.

        Args:
            multi_qudit_gate (Gate | Sequence[Gate]): A multi-qudit gate
                or sequence of gates that starts this layer generator's
                building block. If multiple gates are given as a sequence,
                then a successive layer is built for each gate
                and each location. (Default: IToffoliGate())

            single_qudit_gate (Gate): A single-qudit gate that follows
                `multi_qudit_gate` in the building block. (Default: U3Gate())

        Raises:
            ValueError: If the single-qudit gate's size is not 1.

            ValueError: If there is any radix mismatch between the gates.
        """
        if isinstance(multi_qudit_gates, Gate):
            multi_qudit_gates = [multi_qudit_gates]

        if not all(isinstance(g, Gate) for g in multi_qudit_gates):
            raise TypeError(
                'Expected gate for multi_qudit_gates, got %s.'
                % [type(g) for g in multi_qudit_gates],
            )

        if not isinstance(single_qudit_gate, Gate):
            raise TypeError(
                'Expected gate for single_qudit_gate, got %s.'
                % type(single_qudit_gate),
            )

        if single_qudit_gate.num_qudits != 1:
            raise ValueError(
                'Expected single-qudit gate'
                ', got a gate that acts on %d qudits.'
                % single_qudit_gate.num_qudits,
            )

        sr = single_qudit_gate.radixes[0]
        for mg in multi_qudit_gates:
            if any(r != sr for r in mg.radixes):
                raise ValueError(
                    'Radix mismatch between gates'
                    f': {mg.radixes} !~ {single_qudit_gate.radixes}.',
                )

        self.multi_qudit_gates: list[Gate] = list(multi_qudit_gates)
        self.single_qudit_gate = single_qudit_gate

    def gen_initial_layer(
        self,
        target: UnitaryMatrix | StateVector,
        data: dict[str, Any],
    ) -> Circuit:
        """
        Generate the initial layer, see LayerGenerator for more.

        Raises:
            ValueError: If `target` has a radix mismatch with
                `self.initial_layer_gate`.
        """

        if not isinstance(target, (UnitaryMatrix, StateVector)):
            raise TypeError(
                'Expected unitary or state, got %s.' % type(target),
            )

        for radix in target.radixes:
            if radix != self.single_qudit_gate.radixes[0]:
                raise ValueError(
                    'Radix mismatch between target and single_qudit_gate.',
                )

        init_circuit = Circuit(target.num_qudits, target.radixes)
        for i in range(init_circuit.num_qudits):
            init_circuit.append_gate(self.single_qudit_gate, [i])
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

        if circuit.num_qudits < 2:
            raise ValueError('Cannot expand a single-qudit circuit.')

        # Get the machine model
        model = BasePass.get_model(circuit, data)

        # Generate successors
        successors = []
        for mg in self.multi_qudit_gates:
            width = mg.num_qudits
            for loc in model.coupling_graph.get_subgraphs_of_size(width):
                successor = circuit.copy()
                successor.append_gate(mg, loc)
                for q in loc:
                    successor.append_gate(self.single_qudit_gate, q)
                successors.append(successor)

        return successors
