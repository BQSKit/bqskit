"""This module implements the PhysicalPredicate class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.passes.control.predicate import PassPredicate

_logger = logging.getLogger(__name__)


class PhysicalPredicate(PassPredicate):
    """
    The PhysicalPredicate class.

    The PhysicalPredicate returns true if circuit's coupling graph and gate set
    matches the workflow's machine model.
    """

    def get_truth_value(self, circuit: Circuit, data: dict[str, Any]) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        model = BasePass.get_model(circuit, data)
        for gate in circuit.gate_set:
            if gate not in model.gate_set:
                _logger.debug(f'{gate} not in native set: {model.gate_set}.')
                return False

        if circuit.num_qudits == 1:
            return True

        try:
            placement = BasePass.get_placement(circuit, data)
            subgraph = model.coupling_graph.get_subgraph(placement)
            if not subgraph.is_fully_connected():
                _logger.debug('Qudits are disconnected on the machine model.')
                return False

        except RuntimeError:
            _logger.debug('Qudits are disconnected on the machine model.')
            return False

        return True
