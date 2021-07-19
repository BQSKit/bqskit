"""This module implements the SynthesisPass abstract class."""
from __future__ import annotations

import logging
from abc import abstractmethod
from os.path import exists
from pickle import dump
from typing import Any
from typing import Callable
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passes.util.converttou3 import PauliToU3Pass
from bqskit.compiler.passes.util.converttou3 import VariableToU3Pass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

_logger = logging.getLogger(__name__)

# TODO: Add support for checkpointing other than in QASM format.


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
        checkpoint_dir: str | None = None,
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

            checkpoint_dir (str | None): The path to the directory in which
                checkpoint files should be stored. If it is not provided or
                if the directory does not exist, no checkpoint files will be
                generated.
        """

        self.collection_filter = collection_filter or default_collection_filter
        self.replace_filter = replace_filter or default_replace_filter
        self.checkpoint_dir = checkpoint_dir

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

        if self.checkpoint_dir is not None and not exists(self.checkpoint_dir):
            raise ValueError(
                f'Expected the path to an existing directory; {checkpoint_dir}'
                ' was passed and does not exist.',
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
        structure_list: list[Sequence[int]] = []

        # Synthesize operations
        errors: list[float] = []
        points: list[CircuitPoint] = []
        new_ops: list[Operation] = []
        num_blocks = len(ops_to_syn)

        for block_num, (cycle, op) in enumerate(ops_to_syn):
            sub_numbering = {op.location[i]: i for i in range(op.size)}
            sub_data['machine_model'] = MachineModel(
                len(op.location),
                model.get_subgraph(op.location, sub_numbering),
            )
            structure_list.append([op.location[i] for i in range(op.size)])
            syn_circuit = self.synthesize(op.get_unitary(), sub_data)
            if self.checkpoint_dir is not None:
                save_checkpoint(syn_circuit, self.checkpoint_dir, block_num)
            if self.replace_filter(syn_circuit, op):
                # Calculate errors
                new_utry = syn_circuit.get_unitary()
                old_utry = op.get_unitary()
                error = new_utry.get_distance_from(old_utry)
                errors.append(error)
                points.append(CircuitPoint(cycle, op.location[0]))
                new_ops.append(
                    Operation(
                        CircuitGate(syn_circuit, True),
                        op.location,
                        list(syn_circuit.get_params()),  # TODO: RealVector
                    ),
                )
                _logger.info(
                    f'Error in synthesized CircuitGate {block_num+1} of '
                    f'{num_blocks}: {error}',
                )
        data['synthesispass_error_sum'] = sum(errors)  # TODO: Might be replaced
        _logger.info(
            'Synthesis pass completed. Upper bound on '
            f"circuit error is {data['synthesispass_error_sum']}",
        )
        if self.checkpoint_dir is not None:
            with open(f'{self.checkpoint_dir}/structure.pickle', 'wb') as f:
                dump(structure_list, f)

        circuit.batch_replace(points, new_ops)


def default_collection_filter(op: Operation) -> bool:
    return isinstance(op.gate, (CircuitGate, ConstantUnitaryGate))


def default_replace_filter(circuit: Circuit, op: Operation) -> bool:
    return True


def save_checkpoint(circuit: Circuit, path: str, num: int) -> None:
    circuit_copy = circuit.copy()
    VariableToU3Pass().run(circuit_copy, {})
    PauliToU3Pass().run(circuit_copy, {})
    with open(path + f'/block_{num}', 'w') as f:
        f.write(OPENQASM2Language().encode(circuit_copy))
