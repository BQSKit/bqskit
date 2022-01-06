"""This module implements the SimpleLayerGenerator class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import U3Gate
from bqskit.passes.search.generator import LayerGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
_logger = logging.getLogger(__name__)


class SimpleLayerGenerator(LayerGenerator):
    """
    The SimpleLayerGenerator class.

    Starts a circuit by placing a single-qudit gate on each qudit. Expands a
    circuit by placing a two-qudit building block on all valid links. Each
    building block is composed of a two-qudit gate followed by two single-qudit
    gates.
    """

    def __init__(
        self,
        two_qudit_gate: Gate = CNOTGate(),
        single_qudit_gate_1: Gate = U3Gate(),
        single_qudit_gate_2: Gate | None = None,
        initial_layer_gate: Gate | None = None,
    ) -> None:
        """
        Construct a SimpleLayerGenerator.

        Args:
            two_qudit_gate (Gate): A two-qudit gate that starts this
                layer generator's building block. (Default: CNOTGate())

            single_qudit_gate_1 (Gate): A single-qudit gate that follows
                `two_qudit_gate` in the building block. (Default: U3Gate())

            single_qudit_gate_2 (Gate | None): An alternate single-qudit
                gate to be used as the second single-qudit gate in the
                building block. If left as None, defaults to
                `single_qudit_gate_1`. (Default: None)

            initial_layer_gate (Gate | None): An alternate single-qudit
                gate that creates the initial layer. If left as None,
                defaults to `single_qudit_gate_1`. (Default: None)

        Raises:
            ValueError: If `two_qudit_gate`'s size is not 2, or if any
                of the single-qudit gates' size is not 1.

            ValueError: If `single_qudit_gate_1`'s radix does not match
                the radix of `two_qudit_gate`'s first qudit, or if
                `single_qudit_gate_2`'s radix does not match the radix
                of `two_qudit_gate`'s second qudit.
        """

        if not isinstance(two_qudit_gate, Gate):
            raise TypeError(
                'Expected gate for two_qudit_gate, got %s.'
                % type(two_qudit_gate),
            )

        if two_qudit_gate.num_qudits != 2:
            raise ValueError(
                'Expected two-qudit gate'
                ', got a gate that acts on %d qudits.'
                % two_qudit_gate.num_qudits,
            )

        if not isinstance(single_qudit_gate_1, Gate):
            raise TypeError(
                'Expected gate for single_qudit_gate_1, got %s.'
                % type(single_qudit_gate_1),
            )

        if single_qudit_gate_1.num_qudits != 1:
            raise ValueError(
                'Expected single-qudit gate'
                ', got a gate that acts on %d qudits.'
                % single_qudit_gate_1.num_qudits,
            )

        if single_qudit_gate_2 is None:
            single_qudit_gate_2 = single_qudit_gate_1

        if initial_layer_gate is None:
            initial_layer_gate = single_qudit_gate_1

        if not isinstance(single_qudit_gate_2, Gate):
            raise TypeError(
                'Expected gate for single_qudit_gate_2, got %s.'
                % type(single_qudit_gate_2),
            )

        if single_qudit_gate_2.num_qudits != 1:
            raise ValueError(
                'Expected single-qudit gate'
                ', got a gate that acts on %d qudits.'
                % single_qudit_gate_2.num_qudits,
            )

        if not isinstance(initial_layer_gate, Gate):
            raise TypeError(
                'Expected gate for initial_layer_gate, got %s.'
                % type(initial_layer_gate),
            )

        if initial_layer_gate.num_qudits != 1:
            raise ValueError(
                'Expected single-qudit gate'
                ', got a gate that acts on %d qudits.'
                % initial_layer_gate.num_qudits,
            )

        two_radix_1 = two_qudit_gate.radixes[0]
        two_radix_2 = two_qudit_gate.radixes[1]

        if two_radix_1 != single_qudit_gate_1.radixes[0]:
            raise ValueError(
                'Radix mismatch between two_qudit_gate and single_qudit_gate_1'
                ': %d != %d.'
                % (two_radix_1, single_qudit_gate_1.radixes[0]),
            )

        if two_radix_2 != single_qudit_gate_2.radixes[0]:
            raise ValueError(
                'Radix mismatch between two_qudit_gate and single_qudit_gate_2'
                ': %d != %d.'
                % (two_radix_2, single_qudit_gate_2.radixes[0]),
            )

        self.two_qudit_gate = two_qudit_gate
        self.single_qudit_gate_1 = single_qudit_gate_1
        self.single_qudit_gate_2 = single_qudit_gate_2
        self.initial_layer_gate = initial_layer_gate

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
            if radix != self.initial_layer_gate.radixes[0]:
                raise ValueError(
                    'Radix mismatch between target and initial_layer_gate.',
                )

        init_circuit = Circuit(target.num_qudits, target.radixes)
        for i in range(init_circuit.num_qudits):
            init_circuit.append_gate(self.initial_layer_gate, [i])
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
        for edge in model.coupling_graph:
            successor = circuit.copy()
            successor.append_gate(self.two_qudit_gate, [edge[0], edge[1]])
            successor.append_gate(self.single_qudit_gate_1, edge[0])
            successor.append_gate(self.single_qudit_gate_2, edge[1])
            successors.append(successor)

        return successors
