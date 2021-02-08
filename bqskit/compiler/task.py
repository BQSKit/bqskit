"""
This module implements the CompilationTask class, TaskStatus enum, and
the TaskResult class.

The CompilationTask class describes a compilation problem. These
can be submitted to an engine. The different CompilationTask states
are enumerated in TaskStatus. Once a CompilationTask is completed,
a TaskResult is returned.
"""

from __future__ import annotations
from bqskit.qis.unitarymatrix import UnitaryMatrix

import uuid
from enum import Enum
from typing import Optional, Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit


class TaskStatus(Enum):
    """TaskStatus enum type."""
    ERROR = 0    # The task encountered an error or does not exist.
    WAITING = 1  # The task is waiting in a workqueue.
    RUNNING = 2  # The task is currently running.
    DONE = 3     # The task is finished.

class TaskResult():
    """TaskResult structure."""
    def __init__( self, message: str, circuit: Optional[Circuit] = None ):
        self.message = message
        self.circuit = circuit

class CompilationTask():
    """The CompilationTask class."""

    def __init__(self, input_circuit: Circuit, passes: Sequence[BasePass]) -> None:
        """
        CompilationTask Constructor.

        Args:
            input_circuit (Circuit): The input circuit to be compiled.
        """
        self.task_id = uuid.uuid4()
        self.input_circuit = input_circuit
        self.passes = passes

    @staticmethod
    def synthesize(utry: UnitaryMatrix, method: str) -> CompilationTask:
        pass  # TODO

    @staticmethod
    def optimize(circ: Circuit, method: str) -> CompilationTask:
        pass  # TODO
