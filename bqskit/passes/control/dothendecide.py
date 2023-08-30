"""This module implements the DoThenDecide class."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.workflow import Workflow

if TYPE_CHECKING:
    from typing import Callable

    from bqskit.compiler.passdata import PassData
    from bqskit.compiler.workflow import WorkflowLike
    from bqskit.ir.circuit import Circuit


_logger = logging.getLogger(__name__)


class DoThenDecide(BasePass):
    """
    The DoThenDecide class.

    This is a control pass that executes a workflow and then conditionally
    accepts the resulting circuit or reverts to the original state.
    """

    def __init__(
        self,
        condition: Callable[[Circuit, Circuit], bool],
        workflow: WorkflowLike,
    ) -> None:
        """
        Construct a DoThenDecide.

        Args:
            condition (Callable[[Circuit, Circuit], bool]): The condition
                function that determines if the new circuit
                (second parameter) after running `workflow` should replace
                the original circuit (first parameter). If the condition
                returns True, then replace the original circuit with the
                new one. Note when passing callables to BQSKit passes,
                they need to be defined at the module level (0-indent)
                with a name (no lambdas).

            workflow (WorkflowLike): The pass or passes to execute.
        """
        if not callable(condition):
            bad_type = type(condition)
            msg = f'Expected callable function for condition, got {bad_type}'
            raise TypeError(msg)

        self.condition = condition
        self.workflow = Workflow(workflow)

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        # Store old state
        old_circuit = circuit.copy()
        old_data = data.copy()

        # Execute the workflow
        await self.workflow.run(circuit, data)

        # Evaluate condition
        if self.condition(old_circuit, circuit):
            _logger.info('Accepted circuit.')

        else:
            circuit.become(old_circuit)
            data.become(old_data)
            _logger.info('Rejected circuit.')
