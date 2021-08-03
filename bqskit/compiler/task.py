"""
This module implements the CompilationTask class, TaskException, TaskStatus
enum, and the TaskResult class.

The CompilationTask class describes a compilation problem. These can be
submitted to an engine. The different CompilationTask states are enumerated in
TaskStatus. Once a CompilationTask is completed, a TaskResult is returned. If an
exception occurred during execution, it will be reraised as a TaskException.
"""
from __future__ import annotations

import logging
import uuid
from enum import Enum
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passes.control import ForEachBlockPass
from bqskit.compiler.passes.control.predicates.count import GateCountPredicate
from bqskit.compiler.passes.control.whileloop import WhileLoopPass
from bqskit.compiler.passes.partitioning import GreedyPartitioner
from bqskit.compiler.passes.partitioning.cluster import ClusteringPartitioner
from bqskit.compiler.passes.processing import ScanningGateRemovalPass
from bqskit.compiler.passes.processing import WindowOptimizationPass
from bqskit.compiler.passes.synthesis import LEAPSynthesisPass
from bqskit.compiler.passes.synthesis import QFASTDecompositionPass
from bqskit.compiler.passes.util import UnfoldPass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.qis.unitary.unitarymatrix import UnitaryLike

_logger = logging.getLogger(__name__)


class TaskException(Exception):
    """TaskException exception type."""


class TaskStatus(Enum):
    """TaskStatus enum type."""
    ERROR = 0    # The task encountered an error or does not exist.
    WAITING = 1  # The task is waiting in a workqueue.
    RUNNING = 2  # The task is currently running.
    DONE = 3     # The task is finished.


class TaskResult():
    """TaskResult structure."""

    def __init__(
        self,
        message: str,
        status: TaskStatus,
        circuit: Circuit | None = None,
    ) -> None:
        """TaskResult Constructor."""
        self.message = message
        self.status = status
        self.circuit = circuit

    def get_circuit(self) -> Circuit:
        """Retrieves the circuit result or reraises an error."""
        if self.status == TaskStatus.ERROR:
            raise TaskException(self.message)
        if self.circuit is None:
            raise TaskException('No circuit produced.')
        return self.circuit

    @staticmethod
    def does_not_exist() -> TaskResult:
        """Return a TaskResult with an error message."""
        return TaskResult('Task does not exist.', TaskStatus.ERROR)

    @staticmethod
    def from_circuit(circuit: Circuit) -> TaskResult:
        """Build a simple success result from a circuit."""
        return TaskResult('Success.', TaskStatus.DONE, circuit)


class CompilationTask():
    """The CompilationTask class."""

    def __init__(
        self,
        input_circuit: Circuit,
        passes: Sequence[BasePass],
    ) -> None:
        """
        CompilationTask Constructor.

        Args:
            input_circuit (Circuit): The input circuit to be compiled.

            passes (Sequence[BasePass]): The configured operations to be
                performed on the circuit.
        """
        self.task_id = uuid.uuid4()
        self.input_circuit = input_circuit
        self.passes = passes

    # TODO: Add rebase abilities to default tasks
    @staticmethod
    def synthesis(utry: UnitaryLike) -> CompilationTask:
        """Produces a standard synthesis task for the given unitary."""
        circuit = Circuit.from_unitary(utry)
        num_qudits = circuit.get_size()

        if num_qudits > 8:
            _logger.warning('Synthesis input size is very large.')

        inner_seq = [
            LEAPSynthesisPass(),
            WindowOptimizationPass(),
            ScanningGateRemovalPass(),
        ]

        passes: list[BasePass] = []
        if num_qudits >= 5:
            passes.append(QFASTDecompositionPass())
            passes.append(ForEachBlockPass(inner_seq))
        else:
            passes.extend(inner_seq)

        return CompilationTask(circuit, passes)

    @staticmethod
    def optimize(circuit: Circuit) -> CompilationTask:
        """Produces a standard optimization task for the given circuit."""
        num_qudits = circuit.get_size()

        if num_qudits <= 4:
            return CompilationTask.synthesis(circuit.get_unitary())

        inner_seq = [
            LEAPSynthesisPass(),
            WindowOptimizationPass(),
            ScanningGateRemovalPass(),
        ]

        passes: list[BasePass] = []
        passes.append(GreedyPartitioner(3))
        passes.append(ForEachBlockPass(inner_seq))

        iterative_reopt = WhileLoopPass(
            GateCountPredicate(CNOTGate()),
            [
                ClusteringPartitioner(3, 4),
                ForEachBlockPass(inner_seq),
                UnfoldPass(),
            ],
        )

        passes.append(iterative_reopt)
        return CompilationTask(circuit, passes)
