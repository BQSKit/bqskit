"""This module implements the SynthesisPass abstract class."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.circuit import CircuitPoint
from bqskit.ir.gates.composed.circuitgate import CircuitGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.operation import Operation
from bqskit.qis.unitarymatrix import UnitaryMatrix


class SynthesisPass(BasePass):
    """
    SynthesisPass class.

    The SynthesisPass is a base class that exposes an abstract
    synthesize function. Inherit from this class and implement the
    synthesize function to create a synthesis tool.

    SynthesisPass will iterate through the circuit and call
    the synthesize function on all CircuitGates or ConstantUnitaryGates.
    """

    @abstractmethod
    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:
        """
        Synthesis abstract method to synthesize a UnitaryMatrix into a Circuit.

        Args:
            utry (UnitaryMatrix): The unitary to synthesize.

            data (Dict[str, Any]): Associated data for the pass.
                Can be used to provide auxillary information from
                previous passes. This function should never error based
                on what is in this dictionary.
        """

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """Perform the pass's operation, see BasePass for more info."""

        # Collect synthesizable operations
        ops_to_syn: list[tuple[CircuitPoint, Operation]] = []
        for point, op in circuit.operations_with_points():
            if isinstance(op.gate, (CircuitGate, ConstantUnitaryGate)):
                ops_to_syn.append((point, op))

        # Synthesize operations
        for point, op in ops_to_syn:
            syn_circuit = self.synthesize(op.get_unitary(), data)
            # if isinstance(op.gate, CircuitGate):
            # if len(CircuitGate) > len(syn_circuit):
            # Skip circuits that were synthesized to be longer
            # TODO: different criteria? like depth or 2q gate or cost?
            # continue
            circuit.replace_with_circuit(point.cycle, syn_circuit, op.location)
