"""This test will ensure that the circuit's built by QFAST can be minimized
successfully."""
from __future__ import annotations

from scipy.stats import unitary_group

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import PauliGate
from bqskit.ir.opt.cost.functions import HilbertSchmidtResiduals
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


def test_full_pauli_gate() -> None:
    circuit = Circuit(3)
    circuit.append_gate(PauliGate(3), [0, 1, 2])
    cost = HilbertSchmidtResiduals(circuit, UnitaryMatrix(unitary_group.rvs(8)))
    circuit.minimize(cost)
    assert cost.get_cost(circuit.params) < 1e-6
