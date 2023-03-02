"""This module implements the CompilationTask class."""
from __future__ import annotations

import logging
import uuid
from typing import Any
from typing import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bqskit.compiler.basepass import BasePass
    from bqskit.ir.circuit import Circuit

_logger = logging.getLogger(__name__)


class CompilationTask():
    """
    A complete description of a quantum compilation task.

    The CompilationTask class describes a compilation workflow completely. These
    can be submitted to a BQSKit compiler to be efficiently executed.
    """

    def __init__(self, input: Circuit, workflow: Iterable[BasePass]) -> None:
        """
        Construct a CompilationTask.

        Args:
            input (Circuit): The input circuit to be compiled.

            workflow (Iterable[BasePass]): The configured workflow to be
                performed on the circuit.
        """
        self.task_id = uuid.uuid4()
        self.circuit = input
        self.workflow = workflow

        self.data: dict[str, Any] = {}
        """The task's data for use in BQSKit passes."""

        self.done = False
        """True when the task is complete."""

        self.request_data = False
        """If true, :func:`run` will additionally return the PassData."""

        self.logging_level: int | None = None
        """A general filter on all logging messages in the system."""

        self.max_logging_depth = -1
        """No logging for tasks with more than `max_logging_depth` parents."""

    async def run(self) -> Circuit | tuple[Circuit, dict[str, Any]]:
        """Execute the task."""
        for pass_obj in self.workflow:
            await pass_obj.run(self.circuit, self.data)

        self.done = True

        if not self.request_data:
            return self.circuit

        return self.circuit, self.data
