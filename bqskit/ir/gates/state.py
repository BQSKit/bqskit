"""This module implements the StateGate class."""
from __future__ import annotations

import numpy as np
from bqskit.ir.gate import Gate
from bqskit.qis.state.state import StateVector
from bqskit.qis.state import StateLike
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class StateGate(Gate):
    """
    The StateGate class.

    A StateGate is more-or-less a constant gate that maps the zero
    input state to a specific state. It is undefined what it maps other
    states to.
    """

    def __init__(self, state: StateLike) -> None:
        """
        StateGate Constructor.

        Args:
            state (StateLike): The state this gate should prepare.
        """
        self.state = StateVector(state)
        self._num_qudits = self.state.num_qudits
        self._radixes = self.state.radixes
        self._name = 'StateGate(%s)' % str(self.state)
        self._num_params = 0
    
    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        raise NotImplementedError("State gates do not have unitary representations.")
