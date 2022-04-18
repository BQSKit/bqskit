"""This module implements the FourParamGenerator class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import RXGate
from bqskit.ir.gates import RYGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import U3Gate
from bqskit.passes.search.generator import LayerGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
_logger = logging.getLogger(__name__)


class FourParamGenerator(LayerGenerator):
    """
    The FourParamGenerator class.

    This is an optimized layer generator that uses commutativity rules
    to reduce the number of parameters per block. This also fixes the
    gate set to use cnots, ry, rz, and u3 gates. This is based on the
    following equivalences:

    U--C--U   U--C--Rz--Ry--Rz   U--C--Ry--Rz
       |    ~    |             ~    |
    U--X--U   U--X--Rx--Ry--Rx   U--X--Ry--Rx
    """

    def gen_initial_layer(
        self,
        target: UnitaryMatrix | StateVector,
        data: dict[str, Any],
    ) -> Circuit:
        """
        Generate the initial layer, see LayerGenerator for more.

        Raises:
            ValueError: If `target` is not qubit only.
        """

        if not isinstance(target, (UnitaryMatrix, StateVector)):
            raise TypeError(
                'Expected unitary or state, got %s.' % type(target),
            )

        if not target.is_qubit_only():
            raise ValueError('Cannot generate layers for non-qubit circuits.')

        init_circuit = Circuit(target.num_qudits, target.radixes)
        for i in range(init_circuit.num_qudits):
            init_circuit.append_gate(U3Gate(), [i])
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
            successor.append_gate(CNOTGate(), [edge[0], edge[1]])
            successor.append_gate(RYGate(), edge[0])
            successor.append_gate(RZGate(), edge[0])
            successor.append_gate(RYGate(), edge[1])
            successor.append_gate(RXGate(), edge[1])
            successors.append(successor)

        return successors
