"""This module implements the ParallelDo class."""
from __future__ import annotations

import logging
from typing import Callable
from typing import Iterable
from typing import TYPE_CHECKING

from bqskit.compiler.basepass import _sub_do_work
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.workflow import Workflow
from bqskit.runtime import get_runtime
from bqskit.utils.typing import is_iterable

if TYPE_CHECKING:
    from bqskit.compiler.passdata import PassData
    from bqskit.compiler.workflow import WorkflowLike
    from bqskit.ir.circuit import Circuit


_logger = logging.getLogger(__name__)


class ParallelDo(BasePass):
    """
    The ParallelDo class.

    This is a control pass that executes a sequence of workflows in parallel.
    The branch that is accepted can either be the first to complete or one
    selected by a provided ordering.
    """

    def __init__(
        self,
        pass_sequences: Iterable[WorkflowLike],
        less_than: Callable[[Circuit, Circuit], bool],
        pick_fisrt: bool = False,
    ) -> None:
        """
        Construct a ParallelDo.

        Args:
            pass_sequences (Iterable[WorkflowLike]): The group of workflows
                to run in parallel.

            less_than (Callable[[Circuit, Circuit], bool]): Return True
                if the first circuit is preferred to the second one.
                This will be used to determine which output circuit to
                select.

            pick_first (bool): If true, then the pass will complete as
                soon as one of the workflows finishes and will return
                the first result. Defaults to False.
        """
        if not is_iterable(pass_sequences):
            bad_type = type(pass_sequences)
            raise TypeError(f'Expected sequence of workflows, got {bad_type}.')

        if not callable(less_than):
            bad_type = type(less_than)
            msg = f'Expected callable function for less_than, got {bad_type}'
            raise TypeError(msg)

        self.workflows = [Workflow(p) for p in pass_sequences]
        self.less_than = less_than
        self.pick_first = pick_fisrt

        if len(self.workflows) == 0:
            raise ValueError('Must specify at least one workflow.')

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Running pass sequences in parallel.')

        # Submit jobs to the runtime
        runtime = get_runtime()
        future = runtime.map(
            _sub_do_work,
            self.workflows,
            circuit=circuit,
            data=data,
        )

        # Wait for results
        if self.pick_first:
            circuits_and_ids = await runtime.wait(future)  # Wake on next result
            circuits = [x[1] for x in circuits_and_ids]
            runtime.cancel(future)  # Cancel remaining
        else:
            circuits = await future

        # Find the best result
        best_circ = None
        best_data = None
        for _circ, _data in circuits:
            if best_circ is None or self.less_than(_circ, best_circ):
                best_circ = _circ
                best_data = _data

        if best_circ is None or best_data is None:
            raise RuntimeError('No valid circuit found.')

        # Become best result
        circuit.become(best_circ)
        data.become(best_data)
