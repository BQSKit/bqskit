"""This module implements the ForEachBlockPass class."""
from __future__ import annotations

import logging
from typing import Callable

from bqskit.compiler.basepass import _sub_do_work
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passdata import PassData
from bqskit.compiler.workflow import Workflow
from bqskit.compiler.workflow import WorkflowLike
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.gates.parameterized.pauli import PauliGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.runtime import get_runtime

_logger = logging.getLogger(__name__)


class ForEachBlockPass(BasePass):
    """
    A pass that executes other passes on each block in the circuit.

    This is a control pass that executes a workflow on every block in the
    circuit. This will be done in parallel.
    """

    key = 'ForEachBlockPass_data'
    """The key in data, where block data will be put."""

    def __init__(
        self,
        loop_body: WorkflowLike,
        calculate_error_bound: bool = False,
        collection_filter: Callable[[Operation], bool] | None = None,
        replace_filter: Callable[[Circuit, Operation], bool] | None = None,
        batch_size: int | None = None,
    ) -> None:
        """
        Construct a ForEachBlockPass.

        Args:
            loop_body (WorkflowLike): The workflow to execute on every block.

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

            batch_size (int): (Deprecated).
        """
        if batch_size is not None:
            import warnings
            warnings.warn(
                'Batch size is no longer supported, this warning will'
                ' become an error in a future update.',
                DeprecationWarning,
            )

        self.calculate_error_bound = calculate_error_bound
        self.collection_filter = collection_filter or default_collection_filter
        self.replace_filter = replace_filter or default_replace_filter
        self.workflow = Workflow(loop_body)

        if not callable(self.collection_filter):
            raise TypeError(
                'Expected callable method that maps Operations to booleans for'
                f' collection_filter, got {type(self.collection_filter)}.',
            )

        if not callable(self.replace_filter):
            raise TypeError(
                'Expected callable method that maps Circuit and Operations to'
                f' bools for replace_filter, got {type(self.replace_filter)}.',
            )

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        # Make room in data for block data
        if self.key not in data:
            data[self.key] = []

        # Collect blocks
        blocks: list[tuple[int, Operation]] = []
        for cycle, op in circuit.operations_with_cycles():
            if self.collection_filter(op):
                blocks.append((cycle, op))

        # No blocks, no work
        if len(blocks) == 0:
            data[self.key].append([])
            return

        # Get the machine model
        model = data.model
        coupling_graph = data.connectivity

        # Preprocess blocks
        subcircuits: list[Circuit] = []
        block_datas: list[PassData] = []
        for i, (cycle, op) in enumerate(blocks):

            # Form Subcircuit
            if isinstance(op.gate, CircuitGate):
                subcircuit = op.gate._circuit.copy()
                subcircuit.set_params(op.params)
            else:
                subcircuit = Circuit.from_operation(op)

            # Form Submodel
            subradixes = [circuit.radixes[q] for q in op.location]
            subnumbering = {op.location[i]: i for i in range(len(op.location))}
            submodel = MachineModel(
                len(op.location),
                coupling_graph.get_subgraph(op.location, subnumbering),
                model.gate_set,
                subradixes,
            )

            # Form Subdata
            block_data: PassData = PassData(subcircuit)
            block_data['subnumbering'] = subnumbering
            block_data['model'] = submodel
            block_data['point'] = CircuitPoint(cycle, op.location[0])
            block_data['calculate_error_bound'] = self.calculate_error_bound

            subcircuits.append(subcircuit)
            block_datas.append(block_data)

        # Do the work
        results = await get_runtime().map(
            _sub_do_work,
            [self.workflow] * len(subcircuits),
            subcircuits,
            block_datas,
        )

        # Unpack results
        completed_subcircuits, completed_block_datas = zip(*results)

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

                # Calculate Error
                if self.calculate_error_bound:
                    error_sum += block_data.error
            else:
                block_data['replaced'] = False

        # Replace blocks
        circuit.batch_replace(points, ops)

        # Record block data into pass data
        data[self.key].append(completed_block_datas)

        # Record error
        if self.calculate_error_bound:
            data.error = (1 - ((1 - data.error) * (1 - error_sum)))
            _logger.debug(f'New circuit error is {data.error}.')


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
