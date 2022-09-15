"""This module implements the FillSingleQuditGatesPass class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized.u3 import U3Gate


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

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Completing circuit with single-qudit gates.')
        target = self.get_target(circuit, data)

        complete_circuit = Circuit(circuit.num_qudits, circuit.radixes)

        if target.num_qudits == 1:
            params = U3Gate.calc_params(target)
            complete_circuit.append_gate(U3Gate(), 0, params)
            circuit.become(complete_circuit)
            return

        for q in range(circuit.num_qudits):
            complete_circuit.append_gate(U3Gate(), q)

        for op in circuit:
            if op.num_qudits == 1:
                continue

            complete_circuit.append(op)
            for q in op.location:
                complete_circuit.append_gate(U3Gate(), q)

        dist = 1.0
        while dist > self.success_threshold:
            complete_circuit.instantiate(target)
            dist = complete_circuit.get_unitary().get_distance_from(target, 1)

        circuit.become(complete_circuit)
