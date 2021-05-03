"""This module tests circuit simulation through the get_unitary method."""
from __future__ import annotations

import numpy as np
from numpy.core.numeric import identity

from bqskit.ir.circuit import Circuit
from bqskit.qis.unitary import UnitaryMatrix


def test_toffoli_simulation(
    toffoli_unitary: UnitaryMatrix,
    toffoli_circuit: Circuit,
) -> None:
    calc_unitary = toffoli_circuit.get_unitary()
    assert toffoli_unitary.get_distance_from(calc_unitary) < 1e8

def test_fuzz_simulation(r6_qudit_circuit: Circuit) -> None:
    utry = (r6_qudit_circuit + r6_qudit_circuit.get_dagger()).get_unitary()
    identity = np.identity(r6_qudit_circuit.get_dim())
    assert np.allclose(utry.get_numpy(), identity)
