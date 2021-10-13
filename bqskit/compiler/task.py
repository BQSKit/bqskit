"""This module implements the CompilationTask class."""
from __future__ import annotations

import logging
import uuid
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.passes.control import ForEachBlockPass
from bqskit.passes.control.predicates.count import GateCountPredicate
from bqskit.passes.control.whileloop import WhileLoopPass
from bqskit.passes.partitioning.cluster import ClusteringPartitioner
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.processing import ScanningGateRemovalPass
from bqskit.passes.processing import WindowOptimizationPass
from bqskit.passes.synthesis import QFASTDecompositionPass
from bqskit.passes.synthesis.leap import OptimizedLEAPPass
from bqskit.passes.util import UnfoldPass
from bqskit.qis.unitary.unitarymatrix import UnitaryLike

_logger = logging.getLogger(__name__)


class CompilationTask():
    """
    A complete description of a quantum compilation task.

    The CompilationTask class describes a compilation workflow completely.
    These can be submitted to a BQSKit compiler to be efficiently executed.

    There are static constructors for the most common use cases.
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
        self.input_circuit = input
        self.passes = passes

    @staticmethod
    def synthesize(utry: UnitaryLike) -> CompilationTask:
        """Produces a standard synthesis task for the given unitary."""
        circuit = Circuit.from_unitary(utry)
        num_qudits = circuit.num_qudits

        if num_qudits > 6:
            _logger.warning('Synthesis input size is very large.')

        inner_seq = [
            OptimizedLEAPPass(),
            WindowOptimizationPass(),
            ScanningGateRemovalPass(),
        ]

        passes: list[BasePass] = []
        if num_qudits >= 5:
            passes.append(QFASTDecompositionPass())
            passes.append(ForEachBlockPass(inner_seq))
            passes.append(UnfoldPass())
        else:
            passes.extend(inner_seq)

        return CompilationTask(circuit, passes)

    @staticmethod
    def optimize(circuit: Circuit) -> CompilationTask:
        """Produces a standard optimization task for the given circuit."""
        num_qudits = circuit.num_qudits

        if num_qudits <= 3:
            return CompilationTask.synthesize(circuit.get_unitary())

        inner_seq = [
            OptimizedLEAPPass(),
            ScanningGateRemovalPass(),
        ]

        passes: list[BasePass] = []
        passes.append(QuickPartitioner(3))
        passes.append(ForEachBlockPass(inner_seq))
        passes.append(UnfoldPass())

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
