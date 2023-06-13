"""This module implements the IfThenElsePass class."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.workflow import Workflow
from bqskit.passes.control.predicate import PassPredicate

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.compiler.passdata import PassData
    from bqskit.compiler.workflow import WorkflowLike

_logger = logging.getLogger(__name__)


class IfThenElsePass(BasePass):
    """
    The IfThenElsePass class.

    This is a control pass that conditionally executes a workflow.
    """

    def __init__(
        self,
        condition: PassPredicate,
        on_true: WorkflowLike,
        on_false: WorkflowLike | None = None,
    ) -> None:
        """
        Construct a IfThenElsePass.

        Args:
            condition (PassPredicate): The condition checked.

            on_true (WorkflowLike): The pass or passes to execute if
                `condition` is true.

            on_false (WorkflowLike | None): If specified,
                the pass or passes to execute if `condition` is false.
                Defaults to None, which does is equivalent to a No-Op.
        """

        if not isinstance(condition, PassPredicate):
            raise TypeError('Expected PassPredicate, got %s.' % type(condition))

        self.condition = condition
        self.on_true = Workflow(on_true)
        self.on_false = Workflow(on_false) if on_false is not None else None

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        if self.condition(circuit, data):
            _logger.debug('True branch taken.')
            await self.on_true.run(circuit, data)

        elif self.on_false is not None:
            _logger.debug('False branch taken.')
            await self.on_false.run(circuit, data)
