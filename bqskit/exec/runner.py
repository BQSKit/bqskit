from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from bqskit.exec.results import RunnerResults
from bqskit.ir.circuit import Circuit


class CircuitRunner(ABC):
    """A CircuitRunner is responsible for executing a quantum circuit."""

    @abstractmethod
    def run(self, circuit: Circuit) -> RunnerResults:
        """
        Execute the circuit and return results.

        Args:
            circuit (Circuit): The circuit to run.

        Returns:
            (RunnerResults): The results from executing this circuit.
        """
