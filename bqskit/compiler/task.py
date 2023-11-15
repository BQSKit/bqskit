"""This module implements the CompilationTask class."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from bqskit.compiler.passdata import PassData
from bqskit.compiler.workflow import Workflow

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.compiler.workflow import WorkflowLike


class CompilationTask():
    """
    A complete description of a quantum compilation task.

    The CompilationTask class describes a compilation workflow completely. These
    can be submitted to a BQSKit compiler to be efficiently executed.
    """

    def __init__(self, input: Circuit, workflow: WorkflowLike) -> None:
        """
        Construct a CompilationTask.

        Args:
            input (Circuit): The input circuit to be compiled.

            workflow (WorkflowLike): The configured workflow to be
                performed on the circuit.
        """
        self.task_id = uuid.uuid4()
        self.circuit = input
        self.workflow = Workflow(workflow)

        self.data = PassData(input)
        """The task's data for use in BQSKit passes."""

        self.done = False
        """True when the task is complete."""

        self.request_data = False
        """If true, :func:`run` will additionally return the PassData."""

        self.logging_level: int | None = None
        """A general filter on all logging messages in the system."""

        self.max_logging_depth = -1
        """No logging for tasks with more than `max_logging_depth` parents."""

    async def run(self) -> Circuit | tuple[Circuit, PassData]:
        """Execute the task."""
        await self.workflow.run(self.circuit, self.data)

        if not self.request_data:
            return self.circuit

        return self.circuit, self.data
