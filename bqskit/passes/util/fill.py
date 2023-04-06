"""This module implements the FillSingleQuditGatesPass class."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.point import CircuitPoint


_logger = logging.getLogger(__name__)


class FillSingleQuditGatesPass(BasePass):
    """
    A pass that inserts single-qudit gates around multi-qudit gates.

    This pass will preserve the multi-qudit gates in the circuit and then place
    single-qudit unitary gates around those gates.
    """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Filling circuit with single-qudit gates.')

        complete_circuit = Circuit(circuit.num_qudits, circuit.radixes)
        sq_gate = data.gate_set.get_general_sq_gate()
        id_params = sq_gate.identity_as_params((circuit.radixes[0],))

        for cycle, op in circuit.operations_with_cycles():

            # Single-qudit gates get converting to general gates
            if op.num_qudits == 1:
                params = sq_gate.calc_params(circuit.get_unitary())
                complete_circuit.append_gate(sq_gate, op.location, params)

            else:
                # Multi-qudit gates get added as is
                complete_circuit.append(op)

                # Add additional sq gates where there is none between mq gates
                qudits_needing_single_qudit_gates = set(op.location)
                for p in circuit.next(CircuitPoint(cycle, op.location[0])):
                    if circuit[p].num_qudits == 1:
                        qudits_needing_single_qudit_gates.remove(p.qudit)

                for qudit in qudits_needing_single_qudit_gates:
                    complete_circuit.append_gate(sq_gate, qudit, id_params)

            circuit.become(complete_circuit)
