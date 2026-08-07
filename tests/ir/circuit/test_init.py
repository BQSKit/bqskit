"""This module tests the circuit constructor."""
from __future__ import annotations

import numpy as np
from hypothesis import given

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import HGate
from bqskit.utils.test.strategies import num_qudits_and_radixes


@given(num_qudits_and_radixes())
def test_init(pair: tuple[int, tuple[int, ...]]) -> None:
    num_qudits, radixes = pair
    circuit = Circuit(num_qudits, radixes)
    assert circuit.num_qudits == num_qudits
    assert circuit.radixes == radixes


def test_example() -> None:
    circ = Circuit(2)
    circ.append_gate(HGate(), 0)
    circ.append_gate(CXGate(), (0, 1))
    circ.append_gate(HGate(), 1)
    assert circ.get_unitary() == np.array([
        [0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j],
        [0.5 + 0.j, -0.5 + 0.j, 0.5 + 0.j, -0.5 + 0.j],
        [0.5 + 0.j, 0.5 + 0.j, -0.5 + 0.j, -0.5 + 0.j],
        [-0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j, -0.5 + 0.j],
    ])
    assert circ.get_statevector([1, 0, 0, 0]) == np.array(
        [0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j, -0.5 + 0.j],
    )
