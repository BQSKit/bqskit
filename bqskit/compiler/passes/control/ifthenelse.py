# type: ignore
# TODO: Remove type: ignore, when new mypy comes out with TypeGuards
"""This module implements the IfThenElsePass class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passes.control.predicate import PassPredicate
from bqskit.ir.circuit import Circuit
from bqskit.utils.typing import is_sequence

_logger = logging.getLogger(__name__)


class IfThenElsePass(BasePass):
    """
    The IfThenElsePass class.

    This is a control pass that conditionally executes another pass.
    """

    def __init__(
        self,
        condition: PassPredicate,
        on_true: BasePass | Sequence[BasePass],
        on_false: BasePass | Sequence[BasePass] | None = None,
    ) -> None:
        """
        Construct a IfThenElsePass.

        Args:
            condition (PassPredicate): The condition checked.

            on_true (BasePass | Sequence[BasePass]): The pass or passes
                to execute if `condition` is true.

            on_false (BasePass | Sequence[BasePass] | None): If specified,
                the pass or passes to execute if `condition` is false.
        """

        if not isinstance(condition, PassPredicate):
            raise TypeError('Expected PassPredicate, got %s.' % type(condition))

        if not is_sequence(on_true) and not isinstance(on_true, BasePass):
            raise TypeError(
                'Expected Pass or sequence of Passes, got %s.' % type(on_true),
            )

        if is_sequence(on_true):
            truth_list = [isinstance(elem, BasePass) for elem in on_true]
            if not all(truth_list):
                raise TypeError(
                    'Expected Pass or sequence of Passes, got %s.'
                    % type(on_true[truth_list.index(False)]),
                )

        if on_false is not None:
            if not is_sequence(on_false) and not isinstance(on_false, BasePass):
                raise TypeError(
                    'Expected Pass or sequence of Passes, got %s.'
                    % type(on_false),
                )

            if is_sequence(on_false):
                truth_list = [isinstance(elem, BasePass) for elem in on_false]
                if not all(truth_list):
                    raise TypeError(
                        'Expected Pass or sequence of Passes, got %s.'
                        % type(on_false[truth_list.index(False)]),
                    )

        self.condition = condition
        self.on_true = on_true
        self.on_false = on_false

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """Perform the pass's operation, see BasePass for more info."""
        if self.condition(circuit, data):
            _logger.debug('True branch taken.')
            if is_sequence(self.on_true):
                for true_pass in self.on_true:
                    true_pass.run(circuit, data)
            else:
                self.on_true.run(circuit, data)

        elif self.on_false is not None:
            _logger.debug('False branch taken.')
            if is_sequence(self.on_false):
                for false_pass in self.on_false:
                    false_pass.run(circuit, data)
            else:
                self.on_false.run(circuit, data)
