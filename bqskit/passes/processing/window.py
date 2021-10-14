"""This module implements the WindowOptimizationPass."""
from __future__ import annotations

import logging
from typing import Any
from typing import Callable

import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.passes.synthesis.qsearch import QSearchSynthesisPass
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_iterable

_logger = logging.getLogger(__name__)


class WindowOptimizationPass(BasePass):
    """
    The WindowOptimizationPass class.

    Previously run passes can leave window markers. This pass will select a
    window around that marker and resynthesize, potentially for a depth
    reduction.
    """

    def __init__(
        self,
        window_size: int = 9,
        synthesispass: SynthesisPass = QSearchSynthesisPass(),
        replace_filter: Callable[[Circuit, Circuit], bool] | None = None,
    ) -> None:
        """
        Construct a WindowOptimizationPass.

        Args:
            window_size (int): The size of the window to surround each
                marker. (Default: 9)

            synthesispass (SynthesisPass): The configured synthesis algorithm
                to use during window synthesis.

            replace_filter (Callable[[Circuit, Circuit], bool] | None):
                A predicate that determines if the synthesis result should
                replace the original operation. The first parameter is the
                output from synthesis and the second parameter is the original
                window. If this returns true, the newly synthesized circuit
                will replace the original circuit. Defaults to always replace.

        Raises:
            ValueError: If `window_size` is <= 1.
        """

        if not is_integer(window_size):
            raise TypeError(
                'Expected integer for success_threshold'
                ', got %s' % type(window_size),
            )

        if window_size <= 1:
            raise ValueError(
                'Expected integer greater than 1 for window_size'
                ', got %d.' % window_size,
            )

        if not isinstance(synthesispass, SynthesisPass):
            raise TypeError(
                'Expected SynthesisPass, got %s.' % type(synthesispass),
            )

        self.replace_filter = replace_filter or default_replace_filter

        if not callable(self.replace_filter):
            raise TypeError(
                'Expected callable method for replace_filter'
                ', got %s.' % type(self.replace_filter),
            )

        self.window_size = window_size
        self.synthesispass = synthesispass

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        # Check data for windows markers
        if 'window_markers' not in data:
            _logger.warning('Did not find any window markers.')
            return

        window_markers = data['window_markers']
        _logger.debug('Found window_markers: %s.' % str(window_markers))

        if not is_iterable(window_markers):
            _logger.debug('Invalid type for window_markers.')
            return

        if not all(is_integer(marker) for marker in window_markers):
            _logger.debug('Invalid type for window_markers.')
            return

        # Resynthesis each window
        index_shift = 0
        for marker in window_markers:
            marker -= index_shift

            # Slice window
            begin_cycle = int(marker - self.window_size // 2)
            end_cycle = int(marker + np.ceil(self.window_size / 2))

            if begin_cycle < 0:
                begin_cycle = 0

            if end_cycle > circuit.num_cycles:
                end_cycle = circuit.num_cycles - 1

            window = Circuit(circuit.num_qudits, circuit.radixes)
            window.extend(circuit[begin_cycle:end_cycle])

            _logger.info(
                'Resynthesizing window from cycle '
                f'{begin_cycle} to {end_cycle}.',
            )

            # Synthesize
            utry = window.get_unitary()
            new_window = self.synthesispass.synthesize(utry, data)

            # Replace
            if self.replace_filter(new_window, window):
                _logger.debug('Replacing window with synthesized circuit.')

                actual_window_size = end_cycle - begin_cycle
                for _ in range(actual_window_size):
                    circuit.pop_cycle(begin_cycle)

                circuit.insert_circuit(
                    begin_cycle,
                    new_window,
                    list(range(circuit.num_qudits)),
                )

                index_shift += actual_window_size - new_window.num_cycles


def default_replace_filter(new_circuit: Circuit, og_window: Circuit) -> bool:
    return True
