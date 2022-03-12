"""This module implements the BlockConversionPass."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import ConstantUnitaryGate
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.ir.point import CircuitPoint

_logger = logging.getLogger(__name__)


class BlockConversionPass(BasePass):
    """
    Converts blocks of one type to another type.

    Blocks are either described by a constant or variable unitary gate or as a
    circuit gate. Often during the flow of compilatin we will need to convert
    them from one form to another for a future pass.
    """

    def __init__(
        self,
        convert_target: str,
        convert_variable: bool = True,
        convert_constant: bool = True,
        convert_circuitgates: bool = True,
    ):
        """
        Construct a BlockConversionPass.

        Args:
            convert_target (str): Either `variable` or `constant`.
                Blocks will be converted to the form described here. If
                this is `variable` all gates will be converted to
                `VariableUnitaryGate` s. If this is `constant` blocks
                will be converted to `ConstantUnitaryGate` s. Blocks
                cannot be converted to circuit gates, that can be caried
                out by synthesis.

            convert_variable (bool): If this is true, this will replace
                VariableUnitaryGate's in the circuit with one's specified
                in convert_target.

            convert_constant (bool): If this is true, this will replace
                ConstantUnitaryGate's in the circuit with one's specified
                in convert_target.

            convert_circuitgates (bool): If this is true, this will replace
                CircuitGate's in the circuit with one's specified
                in convert_target. The subcircuit information captured
                in the circuit gate will be lost.
        """
        self.convert_variable = convert_variable
        self.convert_constant = convert_constant
        self.convert_circuitgates = convert_circuitgates

        if convert_target == 'variable':
            self.convert_target = 'variable'
            self.convert_variable = False
        elif convert_target == 'constant':
            self.convert_target = 'constant'
            self.convert_constant = False
        else:
            raise ValueError('Unexpected input for conversion target.')

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        # Variable -> Constant
        if self.convert_variable and self.convert_target == 'constant':
            _logger.debug('Converting variable gates to constant gates.')

            for cycle, op in circuit.operations_with_cycles():
                if isinstance(op.gate, VariableUnitaryGate):
                    cgate = ConstantUnitaryGate(op.get_unitary(), op.radixes)
                    point = CircuitPoint(cycle, op.location[0])
                    circuit.replace_gate(point, cgate, op.location)

        # CircuitGates -> Constant
        if self.convert_circuitgates and self.convert_target == 'constant':
            _logger.debug('Converting circuit gates to constant gates.')

            for cycle, op in circuit.operations_with_cycles():
                if isinstance(op.gate, CircuitGate):
                    cgate = ConstantUnitaryGate(op.get_unitary(), op.radixes)
                    point = CircuitPoint(cycle, op.location[0])
                    circuit.replace_gate(point, cgate, op.location)

        # Constant -> Variable
        if self.convert_constant and self.convert_target == 'variable':
            _logger.debug('Converting constant gates to variable gates.')

            for cycle, op in circuit.operations_with_cycles():
                if isinstance(op.gate, ConstantUnitaryGate):
                    params = VariableUnitaryGate.get_params(op.get_unitary())
                    vgate = VariableUnitaryGate(op.num_qudits, op.radixes)
                    point = CircuitPoint(cycle, op.location[0])
                    circuit.replace_gate(point, vgate, op.location, params)

        # CircuitGates -> Variable
        if self.convert_constant and self.convert_target == 'variable':
            _logger.debug('Converting circuit gates to variable gates.')

            for cycle, op in circuit.operations_with_cycles():
                if isinstance(op.gate, CircuitGate):
                    params = VariableUnitaryGate.get_params(op.get_unitary())
                    vgate = VariableUnitaryGate(op.num_qudits, op.radixes)
                    point = CircuitPoint(cycle, op.location[0])
                    circuit.replace_gate(point, vgate, op.location, params)
