"""This module implements the QFactor class."""
from __future__ import annotations

import logging
from typing import Any
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from bqskitrs import QFactorInstantiatorNative

from bqskit.ir.opt.instantiater import Instantiater
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


_logger = logging.getLogger(__name__)


class QFactor(QFactorInstantiatorNative, Instantiater):
    """The QFactor circuit instantiater."""

    def __new__(cls, **kwargs: dict[str, Any]) -> Any:
        if 'cost_fn_gen' in kwargs:
            del kwargs['cost_fn_gen']
        return super().__new__(cls, **kwargs)

    def instantiate(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
        x0: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Instantiate `circuit`, see Instantiater for more info."""
        return super().instantiate(circuit, target, x0)

    @staticmethod
    def is_capable(circuit: Circuit) -> bool:
        """Return true if the circuit can be instantiated."""
        return all(
            isinstance(gate, LocallyOptimizableUnitary)
            for gate in circuit.gate_set
        )

    @staticmethod
    def get_violation_report(circuit: Circuit) -> str:
        """
        Return a message explaining why `circuit` cannot be instantiated.

        Args:
            circuit (Circuit): Generate a report for this circuit.

        Raises:
            ValueError: If `circuit` can be instantiated with this
                instantiater.
        """

        invalid_gates = {
            gate
            for gate in circuit.gate_set
            if not isinstance(gate, LocallyOptimizableUnitary)
        }

        if len(invalid_gates) == 0:
            raise ValueError('Circuit can be instantiated.')

        return (
            'Cannot instantiate circuit with qfactor'
            ' because the following gates are not locally optimizable: %s.'
            % ', '.join(str(g) for g in invalid_gates)
        )

    @staticmethod
    def get_method_name() -> str:
        """Return the name of this method."""
        return 'qfactor'
