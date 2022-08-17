from __future__ import annotations

from bqskit.ir.gates.measure import MeasurementPlaceholder
from bqskit.ir.lang.qasm2 import OPENQASM2Language


class TestCircuitRemoveAllMeasurements:
    def test_measure_single_bit(self) -> None:
        input = """
            OPENQASM 2.0;
            qreg q[1];
            creg c[1];
            measure q[0] -> c[0];
        """
        circuit = OPENQASM2Language().decode(input)
        assert any(
            isinstance(g, MeasurementPlaceholder)
            for g in circuit.gate_set
        )
        circuit.remove_all_measurements()
        assert not any(
            isinstance(g, MeasurementPlaceholder)
            for g in circuit.gate_set
        )
