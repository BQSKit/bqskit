"""This module implements the ForEachBlockPass class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Callable
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.gates.parameterized.pauli import PauliGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_sequence

_logger = logging.getLogger(__name__)


class ForEachBlockPass(BasePass):
    """
    A pass that executes other passes on each block in the circuit.

    This is a control pass that executes another pass or passes on every block
    in the circuit. This will be done in parallel if run from a compiler.
    """

    key = 'ForEachBlockPass_data'
    """The key in data, where block data will be put."""

    def __init__(
        self,
        loop_body: BasePass | Sequence[BasePass],
        calculate_error_bound: bool = False,
        collection_filter: Callable[[Operation], bool] | None = None,
        replace_filter: Callable[[Circuit, Operation], bool] | None = None,
        batch_size: int = 1,
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

            batch_size (int): The amount of blocks to batch in one job.
                If zero, batch all blocks in one job. (Default: 1).

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

        if not is_integer(batch_size):
            raise TypeError(
                'Expected integer for batch_size'
                f', got {type(batch_size)}',
            )

        if batch_size < 0:
            raise ValueError(
                'Expected nonnegative integer for batch_size'
                f', got {batch_size}',
            )

        self.loop_body = loop_body if is_sequence(loop_body) else [loop_body]
        self.calculate_error_bound = calculate_error_bound
        self.collection_filter = collection_filter or default_collection_filter
        self.replace_filter = replace_filter or default_replace_filter
        self.batch_size = batch_size

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

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        # Make room in data for block data
        if self.key not in data:
            data[self.key] = []

        # Collect blocks
        blocks: list[tuple[int, Operation]] = []
        for cycle, op in circuit.operations_with_cycles():
            if self.collection_filter(op):
                blocks.append((cycle, op))

        # Get the machine model
        model = self.get_model(circuit, data)
        placement = self.get_placement(circuit, data)
        subgraph = model.coupling_graph.get_subgraph(placement)

        # Preprocess blocks
        subcircuits: list[Circuit] = []
        block_datas: list[dict[str, Any]] = []
        for i, (cycle, op) in enumerate(blocks):
            block_data: dict[str, Any] = {}

            # Form Subcircuit
            if isinstance(op.gate, CircuitGate):
                subcircuit = op.gate._circuit.copy()
                subcircuit.set_params(op.params)
            else:
                subcircuit = Circuit.from_operation(op)

            # Form Subtopology
            subradixes = [circuit.radixes[q] for q in op.location]
            subnumbering = {op.location[i]: i for i in range(len(op.location))}
            submodel = MachineModel(
                len(op.location),
                subgraph.get_subgraph(op.location, subnumbering),
                model.gate_set,
                subradixes,
            )

            # Form subdata
            block_data['subnumbering'] = subnumbering
            block_data['machine_model'] = submodel
            block_data['parallel'] = self.in_parallel(data)
            block_data['point'] = CircuitPoint(cycle, op.location[0])
            block_data['calculate_error_bound'] = self.calculate_error_bound

            subcircuits.append(subcircuit)
            block_datas.append(block_data)

        # Group them into batches
        if self.batch_size == 0:
            batched_subcircuits = [subcircuits]
            batched_block_datas = [block_datas]
        else:
            batched_subcircuits = [
                subcircuits[i:i + self.batch_size]
                for i in range(0, len(blocks), self.batch_size)
            ]
            batched_block_datas = [
                block_datas[i:i + self.batch_size]
                for i in range(0, len(blocks), self.batch_size)
            ]

        # Do the work
        results = self.execute(
            data,
            _sub_do_work,
            [self.loop_body] * len(batched_subcircuits),
            batched_subcircuits,
            batched_block_datas,
        )

        # Unpack results
        completed_subcircuits, completed_block_datas = [], []
        for batch in results:
            completed_subcircuits.extend(list(zip(*batch))[0])
            completed_block_datas.extend(list(zip(*batch))[1])

        # Postprocess blocks
        points: list[CircuitPoint] = []
        ops: list[Operation] = []
        error_sum = 0.0
        for i, (cycle, op) in enumerate(blocks):
            subcircuit = completed_subcircuits[i]
            block_data = completed_block_datas[i]

            # Mark Blocks to be Replaced
            if self.replace_filter(subcircuit, op):
                _logger.debug(f'Replacing block {i}.')
                points.append(CircuitPoint(cycle, op.location[0]))
                ops.append(
                    Operation(
                        CircuitGate(subcircuit, True),
                        op.location,
                        subcircuit.params,
                    ),
                )
                block_data['replaced'] = True
            else:
                block_data['replaced'] = False

            # Calculate Error
            if self.calculate_error_bound:
                error_sum += block_data['error']

        # Replace blocks
        circuit.batch_replace(points, ops)

        # Record block data into pass data
        data[self.key].append(completed_block_datas)

        # Record error
        if self.calculate_error_bound:
            if 'error' in data:
                data['error'] *= error_sum
            else:
                data['error'] = error_sum
            _logger.info(f"New circuit error is {data['error']}.")


def default_collection_filter(op: Operation) -> bool:
    return isinstance(
        op.gate, (
            CircuitGate,
            ConstantUnitaryGate,
            VariableUnitaryGate,
            PauliGate,
        ),
    )


def default_replace_filter(circuit: Circuit, op: Operation) -> bool:
    return True


def _sub_do_work(
    loop_body: Sequence[BasePass],
    subcircuits: list[Circuit],
    subdatas: list[dict[str, Any]],
) -> list[tuple[Circuit, dict[str, Any]]]:
    results = []
    for subcircuit, subdata in zip(subcircuits, subdatas):
        if subdata['calculate_error_bound']:
            old_utry = subcircuit.get_unitary()

        for loop_pass in loop_body:
            loop_pass.run(subcircuit, subdata)

        if subdata['calculate_error_bound']:
            new_utry = subcircuit.get_unitary()
            error = new_utry.get_distance_from(old_utry)
            subdata['error'] = error

        results.append((subcircuit, subdata))
    return results
