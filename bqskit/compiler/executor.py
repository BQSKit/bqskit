"""
This module implements the Executor class.

The Executor class is responsible for executing a compilation task.
"""
from __future__ import annotations

import copy
from typing import Any

from bqskit.compiler.task import CompilationTask
from bqskit.ir.circuit import Circuit


class Executor:
    """The Executor class."""

    def __init__(self, task: CompilationTask) -> None:
        """
        Executor Constructor.

        Creates a executor ready to execute the specified task.
        """
        self.task_id = task.task_id
        self.circuit = copy.deepcopy(task.input_circuit)
        self.passes = task.passes
        self.data: dict[str, Any] = {'executor': self}

    def run(self) -> None:
        """Executes the task."""
        for pass_obj in self.passes:
            pass_obj.run(self.circuit, self.data)
        self.done = True

    def get_result(self) -> Circuit:
        """Retrieve result."""
        return self.circuit
