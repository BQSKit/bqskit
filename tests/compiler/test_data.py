from __future__ import annotations

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language


def test_measures_doesnt_error() -> None:
    input = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        measure q[0] -> c[0];
    """
    circuit = OPENQASM2Language().decode(input)
    _ = PassData(circuit)


def test_update_error_mul() -> None:
    data = PassData(Circuit(1))
    assert data.error == 0.0
    data.update_error_mul(0.5)
    assert data.error == 0.5
    data.update_error_mul(0.5)
    assert data.error == 0.75
    data.update_error_mul(0.5)
    assert data.error == 0.875


def test_target_doesnt_get_expanded_on_update() -> None:
    data = PassData(Circuit(64))
    data2 = PassData(Circuit(64))
    data.update(data2)  # Should not crash
