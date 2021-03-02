"""This module implements the SynthesisPass abstract class."""

from abc import abstractmethod
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.gates.composed.circuitgate import CircuitGate
from typing import Any
from bqskit.qis.unitarymatrix import UnitaryMatrix
from bqskit.ir.circuit import Circuit
from bqskit.compiler.basepass import BasePass


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
        ops_to_syn = []
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, [CircuitGate, ConstantUnitaryGate]):
                ops_to_syn.append((cycle, op))
        
        # Synthesize operations
        for cycle, op in ops_to_syn:
            syn_circuit = self.synthesize(op.get_unitary(), data)
            if isinstance(op.gate, CircuitGate):
                if len(CircuitGate) > len(syn_circuit):
                    # Skip circuits that were synthesized to be longer
                    # TODO: different criteria? like depth or 2q gate or cost?
                    continue
            circuit.replace_with_circuit(cycle, syn_circuit, op.location)
