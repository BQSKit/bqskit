"""This module implements the DoThenDecide class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Callable
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.utils.typing import is_sequence

_logger = logging.getLogger(__name__)


class DoThenDecide(BasePass):
    """
    The DoThenDecide class.

    This is a control pass that executes a pass or sequence of passes and then
    conditionally accepts it or reverts to the original state.
    """

    def __init__(
        self,
        condition: Callable[[Circuit, Circuit], bool],
        pass_list: BasePass | Sequence[BasePass],
    ) -> None:
        """
        Construct a DoThenDecide.

        Args:
            condition (Callable[[Circuit, Circuit], bool]): The condition
                function that determines if the new circuit
                (second parameter) after running `pass_list` should replace
                the original circuit (first parameter). If the condition
                returns True, then replaced the original circuit with the
                new one.

            pass_list (BasePass | Sequence[BasePass]): The pass or passes
                to execute.

        Raises:
            ValueError: If a Sequence[BasePass] is given, but it is empty.
        """
        if not is_sequence(pass_list) and not isinstance(pass_list, BasePass):
            raise TypeError(
                'Expected Pass or sequence of Passes, got %s.'
                % type(pass_list),
            )

        if is_sequence(pass_list):
            truth_list = [isinstance(elem, BasePass) for elem in pass_list]
            if not all(truth_list):
                raise TypeError(
                    'Expected Pass or sequence of Passes, got %s.'
                    % type(pass_list[truth_list.index(False)]),
                )
            if len(pass_list) == 0:
                raise ValueError('Expected at least one pass.')

        self.condition = condition
        self.pass_list = pass_list if is_sequence(pass_list) else [pass_list]

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        # Perform Work
        old_circuit = circuit.copy()
        _logger.debug('Pass list executing...')
        for bqskit_pass in self.pass_list:
            bqskit_pass.run(circuit, data)

        if self.condition(old_circuit, circuit):
            _logger.debug('Accepted circuit.')

        else:
            circuit.become(old_circuit)
            _logger.debug('Rejected circuit.')
