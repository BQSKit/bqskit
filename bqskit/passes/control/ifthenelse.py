"""This module implements the IfThenElsePass class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.passes.control.predicate import PassPredicate
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

        Raises:
            ValueError: If a Sequence[BasePass] is given, but it is empty.
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
            if len(on_true) == 0:
                raise ValueError('Expected at least one pass for true branch.')

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
                if len(on_false) == 0:
                    raise ValueError(
                        'Expected at least one pass for false branch.',
                    )

        self.condition = condition
        self.on_true = on_true if is_sequence(on_true) else [on_true]
        self.on_false = on_false if is_sequence(on_false) else [on_false]

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        if self.condition(circuit, data):
            _logger.debug('True branch taken.')
            for branch_pass in self.on_true:
                branch_pass.run(circuit, data)

        elif self.on_false is not None:
            _logger.debug('False branch taken.')
            for branch_pass in self.on_false:
                branch_pass.run(circuit, data)
