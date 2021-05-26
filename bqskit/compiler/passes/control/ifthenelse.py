"""This module implements the IfThenElsePass class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passes.control.predicate import PassPredicate
from bqskit.ir.circuit import Circuit

_logger = logging.getLogger(__name__)


class IfThenElsePass(BasePass):
    """
    The IfThenElsePass class.

    This is a control pass that conditionally executes another pass.
    """

    def __init__(
        self,
        condition: PassPredicate,
        on_true: BasePass,
        on_false: BasePass | None = None,
    ) -> None:
        """
        Construct a IfThenElsePass.

        Args:
            condition (PassPredicate): The condition checked.

            on_true (BasePass): The pass to execute if `condition` is true.

            on_false (BasePass | None): If specified, the pass to execute
                if `condition` is false.
        """

        if not isinstance(condition, PassPredicate):
            raise TypeError('Expected PassPredicate, got %s.' % type(condition))

        if not isinstance(on_true, BasePass):
            raise TypeError('Expected Pass, got %s.' % type(on_true))

        if on_false is not None and not isinstance(on_false, BasePass):
            raise TypeError('Expected Pass, got %s.' % type(on_false))

        self.condition = condition
        self.on_true = on_true
        self.on_false = on_false

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """Perform the pass's operation, see BasePass for more info."""
        if self.condition(circuit, data):
            _logger.debug('True branch taken.')
            self.on_true.run(circuit, data)

        elif self.on_false is not None:
            _logger.debug('False branch taken.')
            self.on_false.run(circuit, data)
