"""
This module implements the Executor class.

The Executor class is responsible for executing a compilation task.
"""
from __future__ import annotations

import copy
from multiprocessing.connection import Connection
from multiprocessing.connection import wait
from typing import Any
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.task import CompilationTask
from bqskit.ir.circuit import Circuit


class Executor:
    """The Executor class."""

    def __init__(
        self, task: CompilationTask,
        workers: Sequence[Connection] = [],
    ) -> None:
        """
        Executor Constructor.

        Creates a executor ready to execute the specified task.
        """
        self.task_id = task.task_id
        self.circuit = copy.deepcopy(task.input_circuit)
        self.passes = task.passes
        self.data: dict[str, Any] = {'executor': self}
        self.workers = workers

    def run(self) -> None:
        """Executes the task."""
        for pass_obj in self.passes:
            pass_obj.run(self.circuit, self.data)
        self.done = True

    def get_result(self) -> Circuit:
        """Retrieve result."""
        return self.circuit

    def parallel_map(
        self,
        passes: Sequence[BasePass],
        circuits_and_data: tuple[Sequence[Circuit], Sequence[dict[str, Any]]],
    ) -> tuple[Sequence[Circuit], Sequence[dict[str, Any]]]:
        """Applies the passes on the circuit and data pairs in parallel."""
        work_list = [
            (passes, circ_and_data)
            for circ_and_data in circuits_and_data
        ]
        last_assigned = -1
        completed = {}

        # Distribute initial round of work
        for worker, work in zip(self.workers, work_list):
            last_assigned += 1
            worker.send((last_assigned, work))

        # Continue until all tasks completed
        while len(completed) != len(work_list):
            for worker in wait(self.workers):
                try:
                    msg = worker.recv()
                except EOFError:
                    self.workers.remove(worker)
                else:
                    completed[msg[0]] = msg[1]
                    if last_assigned < len(work_list) - 1:
                        last_assigned += 1
                        worker.send((last_assigned, work))

        return list(zip(*sorted(completed.items())))[1]
