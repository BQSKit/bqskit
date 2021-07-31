"""This module implements the ForEachBlockPass class."""
from __future__ import annotations

import copy
import logging
from typing import Any
from typing import Callable
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.utils.typing import is_sequence

_logger = logging.getLogger(__name__)


class ForEachBlockPass(BasePass):
    """
    The ForEachBlockPass class.

    This is a control pass that executes another pass or passes on every block
    in the circuit.
    """

    key = 'ForEachBlockPass_data'
    """The key in data, where block data will be put."""

    def __init__(
        self,
        loop_body: BasePass | Sequence[BasePass],
        calculate_error_bound: bool = False,
        collection_filter: Callable[[Operation], bool] | None = None,
        replace_filter: Callable[[Circuit, Operation], bool] | None = None,
    ) -> None:
        """
        Construct a ForEachBlockPass.

        Args:
            loop_body (BasePass | Sequence[BasePass]): The pass or passes
                to execute on every block.

            calculate_error_bound (bool): If set to true, will calculate
                errors on blocks after running `loop_body` on them and
                use these block errors to calculate an upper bound on the
                full circuit error. (Default: False)

            collection_filter (Callable[[Operation], bool] | None):
                A predicate that determines which operations should have
                `loop_body` called on them. Called with each operation
                in the circuit. If this returns true, that operation will
                be formed into an individual circuit and passed through
                `loop_body`. Defaults to all CircuitGates,
                ConstantUnitaryGates, and VariableUnitaryGates.

            replace_filter (Callable[[Circuit, Operation], bool] | None):
                A predicate that determines if the resulting circuit, after
                calling `loop_body` on a block, should replace the original
                operation. Called with the circuit output from `loop_body`
                and the original operation. If this returns true, the
                operation will be replaced with the new circuit.
                Defaults to always replace.

        Raises:
            ValueError: If a Sequence[BasePass] is given, but it is empty.
        """

        if not is_sequence(loop_body) and not isinstance(loop_body, BasePass):
            raise TypeError(
                'Expected Pass or sequence of Passes, got %s.'
                % type(loop_body),
            )

        if is_sequence(loop_body):
            truth_list = [isinstance(elem, BasePass) for elem in loop_body]
            if not all(truth_list):
                raise TypeError(
                    'Expected Pass or sequence of Passes, got %s.'
                    % type(loop_body[truth_list.index(False)]),
                )
            if len(loop_body) == 0:
                raise ValueError('Expected at least one pass.')

        self.loop_body = loop_body if is_sequence(loop_body) else [loop_body]
        self.calculate_error_bound = calculate_error_bound
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

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """Perform the pass's operation, see BasePass for more info."""

        # Make room in data for block data
        if self.key not in data:
            data[self.key] = []
        data[self.key].append([])

        # Collect CircuitGate blocks
        blocks: list[tuple[int, Operation]] = []
        for cycle, op in circuit.operations_with_cycles():
            if self.collection_filter(op):
                blocks.append((cycle, op))

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

        # Go through the blocks
        points: list[CircuitPoint] = []
        ops: list[Operation] = []
        for i, (cycle, op) in enumerate(blocks):
            block_data: dict[str, Any] = {}

            # Form Subcircuit
            if isinstance(op.gate, CircuitGate):
                subcircuit = op.gate._circuit.copy()
                subcircuit.set_params(op.params)
            else:
                subcircuit = Circuit.from_operation(op)

            # Form Subtopology
            subnumbering = {op.location[i]: i for i in range(len(op.location))}
            subdata = {
                'machine_model': MachineModel(
                    len(op.location),
                    model.get_subgraph(op.location, subnumbering),
                ),
            }

            # Record Data Part 1
            block_data['op'] = copy.deepcopy(op)
            block_data['subcircuit_pre'] = subcircuit.copy()

            # Perform Work
            for loop_pass in self.loop_body:
                loop_pass.run(subcircuit, subdata)

            # Record Data Part 2
            block_data['subcircuit_post'] = subcircuit.copy()
            block_data['loop_body_data'] = subdata

            # Calculate Errors
            if self.calculate_error_bound:
                new_utry = subcircuit.get_unitary()
                old_utry = op.get_unitary()
                error = new_utry.get_distance_from(old_utry)
                block_data['error'] = error
                _logger.info(f'Block {i} has error {error}.')

            # Mark Blocks to be Replaced
            if self.replace_filter(subcircuit, op):
                _logger.info(f'Replacing block {i}.')
                points.append(CircuitPoint(cycle, op.location[0]))
                ops.append(
                    Operation(
                        CircuitGate(subcircuit, True),
                        op.location,
                        subcircuit.get_params(),  # type: ignore  # TODO: RealVector  # noqa
                    ),
                )
                block_data['replaced'] = True
            else:
                block_data['replaced'] = False

            # Record block data into pass data
            data[self.key][-1].append(block_data)

        # Replace blocks
        circuit.batch_replace(points, ops)


def default_collection_filter(op: Operation) -> bool:
    return isinstance(
        op.gate, (
            CircuitGate,
            ConstantUnitaryGate,
            VariableUnitaryGate,
        ),
    )


def default_replace_filter(circuit: Circuit, op: Operation) -> bool:
    return True
