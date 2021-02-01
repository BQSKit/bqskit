"""
This module implements the CompilationTask class and TaskStatus enum.

The CompilationTask class describes a compilation problem. These
can be submitted to an engine. The different CompilationTask states
are enumerated in TaskStatus.
"""
import uuid
from enum import Enum
from typing import List

from bqskit.ir.circuit import Circuit


class TaskStatus(Enum):
    """TaskStatus enum type."""
    ERROR = 0    # The task encountered an error or does not exist.
    WAITING = 1  # The task is waiting in a workqueue.
    RUNNING = 2  # The task is currently running.
    DONE = 3     # The task is finished.


class CompilationTask():
    """The CompilationTask class."""

    def __init__(
        self, task_input: Circuit, output_format: str,
        passes: List[str] = [], **kwargs,
    ) -> None:
        """
        CompilationTask Constructor.

        Args:
            task_input (Circuit): The input circuit to be compiled.

            output_format
        """
        self.task_id = uuid.uuid4()
        self.task_input = task_input
        self.output_format = output_format
        self.passes = passes
        self.configuration = kwargs
