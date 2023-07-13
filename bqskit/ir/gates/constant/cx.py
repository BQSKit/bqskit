"""This module implements the CNOTGate/CXGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.qis.unitary import IntegerVector


class CNOTGate(ControlledGate):
    """
    The Controlled-Not or Controlled-X gate for qudits.

    num_levels: (int) = number of levels os single qudit (greater or equal to 2)
    controls: list(int) = list of control levels
    level_1: (int) = first level for shift gate
    level_2: (int) = second level for shift gate
    """

    _qasm_name = 'cx'
    _num_params = 0

    def __init__(self, num_levels: int = 2, controls: IntegerVector = [1], level_1: int = 0, level_2: int = 1):
        super().__init__(
            XGate(num_levels=num_levels, level_1=level_1, level_2=level_2),
            num_levels=num_levels, controls=controls,
        )


CXGate = CNOTGate


class CCXGate(ControlledGate):
    """
    The toffoli gate, equal to an X gate with two controls for qudits.

    num_levels: (int) = number of levels os single qudit (greater or equal to 2)
    controls: list(int) = list of control levels, there are two controls by default
    level_1: (int) = first level for shift gate
    level_2: (int) = second level for shift gate
    """

    _qasm_name = 'ccx'

    def __init__(self, num_levels: int = 2, controls: IntegerVector = [1, 1], level_1: int = 0, level_2: int = 1):
        super().__init__(
            XGate(num_levels=num_levels, level_1=level_1, level_2=level_2),
            num_levels=num_levels, controls=controls,
        )


ToffoliGate = CCXGate
