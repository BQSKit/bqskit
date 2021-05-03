"""This module implements the SynthesisPass abstract class."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any
from typing import Callable

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.circuit import CircuitPoint
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.operation import Operation
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SynthesisPass(BasePass):
    """
    SynthesisPass class.

    The SynthesisPass is a base class that exposes an abstract
    synthesize function. Inherit from this class and implement the
    synthesize function to create a synthesis tool.

    SynthesisPass will iterate through the circuit and call
    the synthesize function on gates that pass the collection filter.
    """

    def __init__(
        self,
        collection_filter: Callable[[Operation], bool] | None = None,
        replace_filter: Callable[[Circuit, Operation], bool] | None = None,
    ):
        """
        SynthesisPass base class constructor.

        Args:
            collection_filter (Callable[[Operation], bool] | None):
                A predicate that determines which operations should be
                synthesized or resynthesized. Called with each operation
                in the circuit. If this returns true, that operation will
                be synthesized by the synthesis pass. Defaults to
                synthesize all CircuitGates and ConstantUnitaryGates.

            replace_filter (Callable[[Circuit, Operation], bool] | None):
                A predicate that determines if the synthesis result should
                replace the original operation. Called with the circuit
                output from synthesis and the original operation. If this
                returns true, the operation will be replaced with the
                synthesized circuit. Defaults to always replace.
        """

        self.collection_filter = collection_filter or default_collection_filter
        self.replace_filter = replace_filter or default_replace_filter

        if not isinstance(self.collection_filter, Callable):
            raise TypeError(
                'Expected callable method that maps Operations to booleans for'
                ' collection_filter, got %s.' % type(self.collection_filter),
            )

        if not isinstance(self.replace_filter, Callable):
            raise TypeError(
                'Expected callable method that maps Circuit and Operations to'
                ' booleans for replace_filter'
                ', got %s.' % type(self.replace_filter),
            )

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

        Note:
            This function should be self-contained and have no side effects.
            This is because it potentially will be called multiple times in
            parallel from one SynthesisPass instance.
        """

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """Perform the pass's operation, see BasePass for more info."""

        # Collect synthesizable operations
        ops_to_syn: list[tuple[CircuitPoint, Operation]] = []
        for point, op in circuit.operations_with_points():
            if self.collection_filter(op):
                ops_to_syn.append((point, op))

        # Synthesize operations
        for point, op in ops_to_syn:  # TODO: Bug point is not invalid on second successful iteration
            syn_circuit = self.synthesize(op.get_unitary(), data)
            if self.replace_filter(syn_circuit, op):
                circuit.replace_with_circuit(point, syn_circuit, op.location)


def default_collection_filter(op: Operation) -> bool:
    return isinstance(op.gate, (CircuitGate, ConstantUnitaryGate))

def default_replace_filter(circuit: Circuit, op: Operation) -> bool:
    return True