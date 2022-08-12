"""This module implements the CompilationTask class."""
from __future__ import annotations

import logging
import uuid
import warnings
from typing import Sequence

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
        warnings.warn(
            'Default task creation is deprecated and will soon be removed.\n'
            'Instead, use the new compile function.',
        )
        circuit = Circuit.from_unitary(utry)
        num_qudits = circuit.num_qudits

        if num_qudits > 6:
            _logger.warning('Synthesis input size is very large.')

        inner_seq = [
            LEAPSynthesisPass(),
            ScanningGateRemovalPass(),
        ]

        passes: list[BasePass] = []
        if num_qudits >= 5:
            passes.append(QFASTDecompositionPass())
            passes.append(
                ForEachBlockPass(
                    inner_seq,
                    replace_filter=less_2q_gates,
                ),
            )
            passes.append(UnfoldPass())
        else:
            passes.extend(inner_seq)

        return CompilationTask(circuit, passes)

    @staticmethod
    def optimize(circuit: Circuit) -> CompilationTask:
        """Produces a standard optimization task for the given circuit."""
        warnings.warn(
            'Default task creation is deprecated and will soon be removed.\n'
            'Instead, use the new compile function.',
        )
        num_qudits = circuit.num_qudits

        if num_qudits <= 3:
            return CompilationTask.synthesize(circuit.get_unitary())

        inner_seq = [
            LEAPSynthesisPass(),
            ScanningGateRemovalPass(),
        ]

        passes: list[BasePass] = []
        passes.append(QuickPartitioner(3))
        passes.append(ForEachBlockPass(inner_seq, replace_filter=less_2q_gates))
        passes.append(UnfoldPass())

        iterative_reopt = WhileLoopPass(
            GateCountPredicate(CNOTGate()),
            [
                ClusteringPartitioner(3, 4),
                ForEachBlockPass(inner_seq, replace_filter=less_2q_gates),
                UnfoldPass(),
            ],
        )

        passes.append(iterative_reopt)
        return CompilationTask(circuit, passes)


def less_2q_gates(circuit: Circuit, op: Operation) -> bool:
    """Replace `circuit' with `op` if has less 2 qubit gates."""
    if not isinstance(op, CircuitGate):
        return True
    og_num_2q_gate = len([op for op in op._circuit if op.num_qudits >= 2])
    new_num_2q_gate = len([op for op in circuit if op.num_qudits >= 2])
    return new_num_2q_gate > og_num_2q_gate
