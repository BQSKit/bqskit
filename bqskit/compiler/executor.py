"""
This module implements the Executor class.

The Executor class is responsible for executing a compilation task.
"""
from __future__ import annotations

from bqskit.compiler.task import CompilationTask
from bqskit.ir.circuit import Circuit


class Executor:
    """The Executor class."""

    def __init__(self, task: CompilationTask) -> None:
        """
        Executor Constructor.

        Creates a executor ready to execute the specified task.
        """

        self.task = task
        # TODO: Initialize passes with configurations

    def run(self) -> None:
        pass

    def get_result(self) -> Circuit:
        pass
