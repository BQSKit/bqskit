from __future__ import annotations

from typing import cast
from typing import Sequence

import numpy as np

from bqskit.exec.results import RunnerResults
from bqskit.exec.runner import CircuitRunner
from bqskit.ir.circuit import Circuit


class SimulationRunner(CircuitRunner):
    """Simulate a circuit."""

    def run(self, circuit: Circuit) -> RunnerResults:
        """Execute the circuit, see CircuitRunner.run for more info."""
        state = np.reshape(circuit.get_unitary()[:, 0], (-1))
        probs = np.square(np.abs(state), dtype=np.float64)
        probs = cast(Sequence[float], probs)
        return RunnerResults(circuit.num_qudits, circuit.radixes, probs)
