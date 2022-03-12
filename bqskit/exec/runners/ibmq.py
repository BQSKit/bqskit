from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.providers.ibmq import IBMQBackend

from bqskit.exec.results import RunnerResults
from bqskit.exec.runner import CircuitRunner
from bqskit.ir.circuit import Circuit


class IBMQRunner(CircuitRunner):
    """Simulate a circuit."""

    def __init__(self, backend: IBMQBackend) -> None:
        """Setup an IBMQRunner to execute circuits on `backend`."""
        self.backend = backend

    def run(self, circuit: Circuit) -> RunnerResults:
        """Execute the circuit, see CircuitRunner.run for more info."""
        # 1. Check circuit and self.backend are compatible
        # TODO

        # 2. Convert to Qiskit IR
        qiskit_circ = QuantumCircuit.from_qasm_str(circuit.to('qasm'))
        qiskit_circ.measure_all()

        # 3. Run circuit
        result = self.backend.run(qiskit_circ).result()
        shots = result.results[0].shots
        probs = [0.0 for i in range(2 ** circuit.num_qudits)]
        for bit_str, count in result.get_counts().items():
            probs[int(bit_str, 2)] = count / shots

        return RunnerResults(circuit.num_qudits, circuit.radixes, probs)
