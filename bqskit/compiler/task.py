"""This module implements the CompilationTask class."""
from __future__ import annotations

import logging
import uuid
import warnings
from typing import Any, Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.operation import Operation
from bqskit.passes.control import ForEachBlockPass
from bqskit.passes.control.predicates.count import GateCountPredicate
from bqskit.passes.control.whileloop import WhileLoopPass
from bqskit.passes.partitioning.cluster import ClusteringPartitioner
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.processing import ScanningGateRemovalPass
from bqskit.passes.synthesis import QFASTDecompositionPass
from bqskit.passes.synthesis.leap import LEAPSynthesisPass
from bqskit.passes.util import UnfoldPass
from bqskit.qis.unitary.unitarymatrix import UnitaryLike

_logger = logging.getLogger(__name__)


class CompilationTask():
    """
    A complete description of a quantum compilation task.

    The CompilationTask class describes a compilation workflow completely. These
    can be submitted to a BQSKit compiler to be efficiently executed.
    """

    def __init__(self, input: Circuit, passes: Sequence[BasePass]) -> None:
        """
        Construct a CompilationTask.

        Args:
            input (Circuit): The input circuit to be compiled.

            passes (Sequence[BasePass]): The configured operations to be
                performed on the circuit.
        """
        self.task_id = uuid.uuid4()
        self.circuit = input
        self.passes = passes
        self.data = {}
        self.requested_keys = []
        self.logging_level = 30
        self.max_logging_depth = -1

    async def run(self) -> Circuit | tuple[Circuit, dict[str, Any]]:
        """Execute the task."""
        for pass_obj in self.passes:
            await pass_obj.run(self.circuit, self.data)
        self.done = True

        if len(self.requested_keys) == 0:
            return self.circuit

        requested_data = {
            k: self.data[k]
            for k in self.requested_keys
            if k in self.data
        }
        return self.circuit, requested_data
    
    def request_key(self, key: str) -> None:
        """Ask the task to also return `key` from the compilation data."""
        self.requested_keys.append(key)

    def set_max_logging_depth(self, max_depth: int) -> None:
        """Restrict logging for tasks with more than `max_depth` parents."""
        self.max_logging_depth = max_depth
