"""This module implements the ManyQuditGatesPredicate class."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bqskit.passes.control.predicate import PassPredicate

if TYPE_CHECKING:
    from bqskit.compiler.passdata import PassData
    from bqskit.ir.circuit import Circuit

_logger = logging.getLogger(__name__)


class ManyQuditGatesPredicate(PassPredicate):
    """Check if many-qudit gates are in the circuit and/or model."""

    def __init__(
        self,
        check_circuit: bool = True,
        check_model: bool = True,
    ) -> None:
        """
        Construct a ManyQuditGatesPredicate.

        Args:
            check_circuit (bool): Whether to check the circuit for many-qudit
                gates. (Default: True)

            check_model (bool): Whether to check the model for many-qudit
                gates. (Default: True)

        Raises:
            ValueError: If both check_circuit and check_model are False.
        """
        if not isinstance(check_circuit, bool):
            raise TypeError(
                f'Expected bool for check_circuit, got {type(check_circuit)}.',
            )

        if not isinstance(check_model, bool):
            raise TypeError(
                f'Expected bool for check_model, got {type(check_model)}.',
            )

        if not check_circuit and not check_model:
            raise ValueError(
                'At least one of check_circuit or check_model must be True.',
            )

        self.check_circuit = check_circuit
        self.check_model = check_model

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        if self.check_circuit:
            for gate in circuit.gate_set_no_blocks:
                if gate.num_qudits > 2:
                    _logger.debug(f'Found many-qudit gate in circuit: {gate}.')
                    return True

        if self.check_model:
            for gate in data.gate_set:
                if gate.num_qudits > 2:
                    _logger.debug(f'Found many-qudit gate in model: {gate}.')
                    return True

        return False
