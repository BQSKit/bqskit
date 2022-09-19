"""This module implements the PhysicalPredicate class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.passes.control.predicate import PassPredicate

_logger = logging.getLogger(__name__)


class SinglePhysicalPredicate(PassPredicate):
    """
    The SinglePhysicalPredicate class.

    The SinglePhysicalPredicate returns true if circuit's single-qudit gates are
    in the native gate set.
    """

    def get_truth_value(self, circuit: Circuit, data: dict[str, Any]) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        model = BasePass.get_model(circuit, data)
        for gate in circuit.gate_set:
            if gate.num_qudits > 1:
                continue
            if gate not in model.gate_set:
                _logger.debug(f'{gate} not in native set: {model.gate_set}.')
                return False

        return True
