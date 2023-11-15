"""This module implements the SinglePhysicalPredicate class."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bqskit.ir.gates.constant.sx import SXGate
from bqskit.ir.gates.generalgate import GeneralGate
from bqskit.ir.gates.parameterized.rx import RXGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gates.parameterized.u1 import U1Gate
from bqskit.passes.control.predicate import PassPredicate

if TYPE_CHECKING:
    from bqskit.compiler.passdata import PassData
    from bqskit.ir.circuit import Circuit

_logger = logging.getLogger(__name__)


class SinglePhysicalPredicate(PassPredicate):
    """
    The SinglePhysicalPredicate class.

    The SinglePhysicalPredicate returns true if circuit's single-qudit gates are
    in the native gate set.
    """

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        for gate in circuit.gate_set:
            if gate.num_qudits > 1:
                continue
            if gate not in data.gate_set:
                _logger.debug(f'{gate} not in native set: {data.gate_set}.')
                return False

        return True


class NoSingleQuditGatesInModel(PassPredicate):
    """A predicate that checks if the model has single qudit gates."""

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        return len(data.gate_set.single_qudit_gates) == 0


class HasGeneralSingleQuditGate(PassPredicate):
    """A predicate that checks if the model has a general single qudit gate."""

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        return any(
            isinstance(g, GeneralGate)
            for g in data.gate_set.single_qudit_gates
        )


class ZXGatePredicate(PassPredicate):
    """A predicate that checks if the model has a RZ or U1 and SX or RX gate."""

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        return (
            (
                RZGate() in data.gate_set.single_qudit_gates
                or U1Gate() in data.gate_set.single_qudit_gates
            ) and (
                SXGate() in data.gate_set.single_qudit_gates
                or RXGate() in data.gate_set.single_qudit_gates
            )
        )


class AllConstantSingleQuditGates(PassPredicate):
    """A predicate that checks if all single qudit gates are constant."""

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        return all(g.is_constant() for g in data.gate_set.single_qudit_gates)
