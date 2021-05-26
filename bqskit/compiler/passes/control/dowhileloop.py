"""This module implements the DoWhileLoopPass class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passes.control.predicate import PassPredicate
from bqskit.ir.circuit import Circuit

_logger = logging.getLogger(__name__)


class DoWhileLoopPass(BasePass):
    """
    The DoWhileLoopPass class.

    This is a control pass that executes a pass and then conditionally executes
    it again in a loop.
    """

    def __init__(
        self,
        condition: PassPredicate,
        loop_body: BasePass,
    ) -> None:
        """
        Construct a DoWhileLoopPass.

        Args:
            condition (PassPredicate): The condition checked.

            loop_body (BasePass): The pass to execute while `condition`
                is true.
        """

        if not isinstance(condition, PassPredicate):
            raise TypeError('Expected PassPredicate, got %s.' % type(condition))

        if not isinstance(loop_body, BasePass):
            raise TypeError('Expected Pass, got %s.' % type(loop_body))

        self.condition = condition
        self.loop_body = loop_body

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """Perform the pass's operation, see BasePass for more info."""
        self.loop_body.run(circuit, data)

        while self.condition(circuit, data):
            _logger.debug('Loop body executing...')
            self.loop_body.run(circuit, data)
