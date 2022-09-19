"""This module implements the Executor class."""
from __future__ import annotations

from typing import TYPE_CHECKING

from bqskit.compiler.task import CompilationTask
if TYPE_CHECKING:
    from typing import Any
    from bqskit.ir.circuit import Circuit


class Executor:
    """An Executor is responsible for executing a CompilationTask."""

    def __init__(self, task: CompilationTask) -> None:
        """
        Construct an Executor.

        Creates a executor ready to execute the specified task.

        Args:
            task (CompilationTask): The task to execute.
        """

        if not isinstance(task, CompilationTask):
            raise TypeError(f'Expected a CompilationTask, got {type(task)}.')

        self.task_id = task.task_id
        self.circuit = task.input_circuit
        self.passes = task.passes
        self.data: dict[str, Any] = {'parallel': True}
        self.done = False

    def run(self) -> tuple[Circuit, dict[str, Any]]:
        """Execute the task."""
        for pass_obj in self.passes:
            pass_obj.run(self.circuit, self.data)
        self.done = True
        return self.circuit, self.data
