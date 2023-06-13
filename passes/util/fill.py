"""This module implements the FillSingleQuditGatesPass class."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.qubit.parameterized.u3 import U3Gate
from bqskit.ir.gates.qubit.parameterized.u8 import U8Gate
from bqskit.ir.gates.qubit.parameterized.unitary import VariableUnitaryGate


_logger = logging.getLogger(__name__)


class FillSingleQuditGatesPass(BasePass):
    """A pass that inserts single-qudit gates around multi-qudit gates."""

    def __init__(self, success_threshold: float = 1e-10):
        """
        Construct a FillSingleQuditGatesPass.

        Args:
            success_threshold (bool): Reinstantiate the new filled circuit
                to be within this distance from initial starting circuit.
        """
        self.success_threshold = success_threshold

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Completing circuit with single-qudit gates.')
        target = data.target

        complete_circuit = Circuit(circuit.num_qudits, circuit.radixes)

        if target.num_qudits == 1 and circuit.radixes[0] == 2:
            params = U3Gate.calc_params(circuit.get_unitary())
            complete_circuit.append_gate(U3Gate(), 0, params)
            circuit.become(complete_circuit)
            return

        for q in range(circuit.num_qudits):
            radix = circuit.radixes[q]
            if radix == 2:
                complete_circuit.append_gate(U3Gate(), q)
            elif radix == 3:
                complete_circuit.append_gate(U8Gate(), q)
            else:
                complete_circuit.append_gate(VariableUnitaryGate(1), q)

        for op in circuit:
            if op.num_qudits == 1:
                continue

            complete_circuit.append(op)
            for q in op.location:
                radix = circuit.radixes[q]
                if radix == 2:
                    complete_circuit.append_gate(U3Gate(), q)
                elif radix == 3:
                    complete_circuit.append_gate(U8Gate(), q)
                else:
                    complete_circuit.append_gate(VariableUnitaryGate(1), q)

        dist = 1.0
        for i in range(10):
            complete_circuit.instantiate(target)
            dist = complete_circuit.get_unitary().get_distance_from(target, 1)  # type: ignore  # noqa
            # TODO: State update

        if dist <= self.success_threshold:
            circuit.become(complete_circuit)
        else:
            _logger.warning('Unable to instantiate completed circuit.')
