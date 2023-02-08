"""This module implements the BlockAnalysisPass."""
from __future__ import annotations

import logging
from typing import Any
from typing import Callable

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CircuitGate

_logger = logging.getLogger(__name__)


def gate_count(block: Circuit) -> dict[Gate, int]:
    return block.gate_counts


class BlockAnalysisPass(BasePass):
    """
    Scaffold to analyze blocks of type CircuitGate.

    A Circuit may be partitioned into blocks, each of which may have very
    different properties. This pass allows for blocks in a circuit to be
    analyzed and compared quickly.

    This pass scans through a circuit, and stores the results of the scan in a
    list variable called `results`.
    """

    def __init__(
        self,
        analysis_function: Callable[[Circuit], Any] = gate_count,
        filter_function: Callable[[Circuit], Any] = lambda x: x,
    ):
        """
        Construct a BlockAnalysisPass.

        Args:
            analysis_function (callable): A function to call on each block of
                type `CircuitGate` in a circuit. Note that this function
                assumes the argument is given as a `Circuit`.
                (Default: Circuit.gate_counts)

            filter_function (callable | None): A function that will filter out
                blocks that meet some criteron. If `filter_function` returns
                False for some block, it will be filtered out of the analysis.
                It is assumed that this function takes a Circuit as an arugment.
                (Default: lambda x: x)
        """
        self.analysis_function = analysis_function
        self.filter_function = filter_function
        self.results: list[Any] = []

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        if len(self.results) != 0:
            _logger.debug('Clearing BlockAnalysisPass results...')
            self.results = []

        for op in circuit:
            if type(op.gate) != CircuitGate:
                continue

            # Convert to subcircuit
            block = Circuit(op.num_qudits)
            block.append_gate(
                op.gate,
                [_ for _ in range(op.num_qudits)],
                op.params,
            )

            block.unfold_all()
            if not self.filter_function(block):
                continue

            result = self.analysis_function(block)
            self.results.append(result)
