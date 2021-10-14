"""This module implements the WhileLoopPass class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.passes.control.predicate import PassPredicate
from bqskit.utils.typing import is_sequence

_logger = logging.getLogger(__name__)


class WhileLoopPass(BasePass):
    """
    The WhileLoopPass class.

    This is a control pass that conditionally executes another pass in a loop.
    """

    def __init__(
        self,
        condition: PassPredicate,
        loop_body: BasePass | Sequence[BasePass],
    ) -> None:
        """
        Construct a WhileLoopPass.

        Args:
            condition (PassPredicate): The condition checked.

            loop_body (BasePass | Sequence[BasePass]): The pass or passes
                to execute while `condition` is true.

        Raises:
            ValueError: If a Sequence[BasePass] is given, but it is empty.
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
            if len(loop_body) == 0:
                raise ValueError('Expected at least one pass.')

        self.condition = condition
        self.loop_body = loop_body if is_sequence(loop_body) else [loop_body]

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        while self.condition(circuit, data):
            _logger.debug('Loop body executing...')
            for loop_pass in self.loop_body:
                loop_pass.run(circuit, data)
