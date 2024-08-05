from __future__ import annotations

from bqskit.ir.circuit import Circuit


def test_circuit_dim_overflow() -> None:
    c = Circuit(1024)
    assert c.dim != 0
