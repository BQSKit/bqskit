# type: ignore
# TODO: Remove type: ignore, when new mypy comes out with TypeGuards
"""This module implements the DoWhileLoopPass class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passes.control.predicate import PassPredicate
from bqskit.ir.circuit import Circuit
from bqskit.utils.typing import is_sequence

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
        loop_body: BasePass | Sequence[BasePass],
    ) -> None:
        """
        Construct a DoWhileLoopPass.

        Args:
            condition (PassPredicate): The condition checked.

            loop_body (BasePass | Sequence[BasePass]): The pass or passes
                to execute while `condition` is true.
        """

        if not isinstance(condition, PassPredicate):
            raise TypeError('Expected PassPredicate, got %s.' % type(condition))

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

        self.condition = condition
        self.loop_body = loop_body

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """Perform the pass's operation, see BasePass for more info."""
        self.loop_body.run(circuit, data)

        while self.condition(circuit, data):
            _logger.debug('Loop body executing...')
            if is_sequence(self.loop_body):
                for loop_pass in self.loop_body:
                    loop_pass.run(circuit, data)
            else:
                self.loop_body.run(circuit, data)
