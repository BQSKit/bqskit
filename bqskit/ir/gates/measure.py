"""This module implements the MeasurementPlaceholder class."""
from __future__ import annotations

from bqskit.ir.gate import Gate
from bqskit.ir.location import CircuitLocation
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class MeasurementPlaceholder(Gate):
    """Pseudogate to hold measurement information."""

    def __init__(
        self,
        classical_regs: list[tuple[str, int]],
        measurements: dict[int, tuple[str, int]],
    ) -> None:
        """
        Construct a MeasurementPlaceholder.

        Args:
            classical_regs (list[tuple[str, int]]): A list of classical
                register descriptors. Each one is given as a tuple containing
                its name and size.

            measurements (dict[int, tuple[str, int]]): A list of measurements
                made. Measurements are given as a map of qudit index to
                a tuple containing the classical register's name and index.
        """
        self._name = 'measurement'
        self._num_qudits = len(measurements)
        self._radixes = tuple([2] * self._num_qudits)
        self._num_params = 0
        self.classical_regs = classical_regs
        self.measurements = measurements

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        raise RuntimeError(
            'Cannot compute unitary for a measurement placeholder.'
            ' Consider removing measurements either by calling'
            ' `circuit.remove_measurements()` or by using the'
            ' ExtractMeasurements and RestoreMeasurements passes.',
        )

    def get_qasm_gate_def(self) -> str:
        """Declares the classical registers."""
        ret = ''
        for name, size in self.classical_regs:
            ret += f'creg {name}[{size}];\n'
        return ret

    def get_qasm(self, params: RealVector, location: CircuitLocation) -> str:
        ret = ''
        for qudit, (name, index) in self.measurements.items():
            ret += f'measure q[{qudit}] -> {name}[{index}];\n'
        return ret

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MeasurementPlaceholder)
            and other.measurements == self.measurements
            and other.classical_regs == self.classical_regs
        )

    def __hash__(self) -> int:
        return hash(tuple(self.classical_regs))
