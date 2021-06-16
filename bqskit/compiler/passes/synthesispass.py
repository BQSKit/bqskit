"""This module implements the SynthesisPass abstract class."""
from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any
from typing import Callable

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.operation import Operation
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

_logger = logging.getLogger(__name__)


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

        if not callable(self.collection_filter):
            raise TypeError(
                'Expected callable method that maps Operations to booleans for'
                ' collection_filter, got %s.' % type(self.collection_filter),
            )

        if not callable(self.replace_filter):
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
        ops_to_syn: list[tuple[int, Operation]] = []
        for cycle, op in circuit.operations_with_cycles():
            if self.collection_filter(op):
                ops_to_syn.append((cycle, op))

        # If a MachineModel is provided in the data dict, it will be used.
        # Otherwise all-to-all connectivity is assumed.
        model = None
        if 'machine_model' in data:
            model = data['machine_model']
        if (
            not isinstance(model, MachineModel)
            or model.num_qudits < circuit.get_size()
        ):
            _logger.warning(
                'MachineModel not specified or invalid;'
                ' defaulting to all-to-all.',
            )
            model = MachineModel(circuit.get_size())

        sub_data = data.copy()

        # Synthesize operations
        errors: list[float] = []
        for cycle, op in ops_to_syn:
<<<<<<< HEAD
            sub_numbering = {op.location[i]: i for i in range(len(op.location))}
            sub_data['machine_model'] = MachineModel(
                len(op.location),
                model.get_subgraph(op.location, sub_numbering),
            )

            syn_circuit = self.synthesize(op.get_unitary(), sub_data)
=======
            syn_circuit = self.synthesize(op.get_unitary(), data)
>>>>>>> 98145a309621568f04dfa6c405c65a29304cf2d8

            if self.replace_filter(syn_circuit, op):
                # Calculate errors
                new_utry = syn_circuit.get_unitary()
                old_utry = op.get_unitary()
                errors.append(new_utry.get_distance_from(old_utry))

                circuit.replace_gate(
                    (cycle, op.location[0]),
                    CircuitGate(syn_circuit, True),
                    op.location,
                    list(syn_circuit.get_params()),  # TODO: RealVector
                )

        data['synthesispass_error_sum'] = sum(errors)  # TODO: Might be replaced
        _logger.info(
            'Synthesis pass completed. Upper bound on '
            f"circuit error is {data['synthesispass_error_sum']}",
        )


def default_collection_filter(op: Operation) -> bool:
    return isinstance(op.gate, (CircuitGate, ConstantUnitaryGate))


def default_replace_filter(circuit: Circuit, op: Operation) -> bool:
    return True
