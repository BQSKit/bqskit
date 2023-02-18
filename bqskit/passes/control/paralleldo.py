"""This module implements the ParallelDo class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Callable
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.runtime import get_runtime
from bqskit.utils.typing import is_sequence

_logger = logging.getLogger(__name__)


class ParallelDo(BasePass):
    """
    The ParallelDo class.

    This is a control pass that executes a group of pass sequences in parallel.
    """

    def __init__(
        self,
        pass_sequences: Sequence[BasePass | Sequence[BasePass]],
        less_than: Callable[[Circuit, Circuit], bool],
        pick_fisrt: bool = False,
    ) -> None:
        """
        Construct a ParallelDo.

        Args:
            pass_sequences (Sequence[BasePass | Sequence[BasePass]]):
                The group of pass sequences to run in parallel.

            less_than (Callable[[Circuit, Circuit], bool]): Return True
                if the first circuit is preferred to the second one.
                This will be used to determine which output circuit to
                select.

            pick_first (bool): If true, then the pass will complete as
                soon as one of the subsequences finishes and will return
                the first result.

        Raises:
            ValueError: If a Sequence[BasePass] is given, but it is empty.
        """
        if not is_sequence(pass_sequences):
            raise TypeError(
                'Expected sequence of sequences of passes, got %s.'
                % type(pass_sequences),
            )

        self.pass_seqs = []
        for ps in pass_sequences:
            if not is_sequence(ps) and not isinstance(ps, BasePass):
                raise TypeError(
                    'Expected Pass or sequence of Passes, got %s.'
                    % type(ps),
                )

            if is_sequence(ps):
                truth_list = [isinstance(elem, BasePass) for elem in ps]
                if not all(truth_list):
                    raise TypeError(
                        'Expected Pass or sequence of Passes, got %s.'
                        % type(ps[truth_list.index(False)]),
                    )
                if len(ps) == 0:
                    raise ValueError('Expected at least one pass.')

            self.pass_seqs.append(ps if is_sequence(ps) else [ps])

        self.less_than = less_than
        self.pick_first = pick_fisrt

    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Running pass sequences in parallel.')

        # Submit jobs to the runtime
        runtime = get_runtime()
        future = runtime.map(
            _sub_do_work,
            self.pass_seqs,
            circuit = circuit,
            data = data,
        )

        # Wait for results
        if self.pick_first:
            circuits_and_ids = await runtime.wait(future) # Wake on next result
            circuits = [x[1] for x in circuits_and_ids]
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
        data.update(best_data)


async def _sub_do_work(
    loop_body: Sequence[BasePass],
    circuit: Circuit,
    data: dict[str, Any],
) -> tuple[Circuit, dict[str, Any]]:
    """Execute a sequence of passes on circuit"""
    # TODO: Move to BasePass
    for loop_pass in loop_body:
        await loop_pass.run(circuit, data)
    return circuit, data
