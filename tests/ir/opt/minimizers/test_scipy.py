"""This module tests the ScipyMinimizer class."""
from __future__ import annotations

from bqskit.ir import Circuit
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import SXGate
from bqskit.ir.opt import HilbertSchmidtCostGenerator
from bqskit.ir.opt import ScipyMinimizer
from bqskit.qis.unitary import UnitaryMatrix


def test_scipy_minimize_sq_circuit() -> None:
    circuit = Circuit(1)
    circuit.append_gate(RZGate(), 0)
    circuit.append_gate(SXGate(), 0)
    circuit.append_gate(RZGate(), 0)
    circuit.append_gate(SXGate(), 0)
    circuit.append_gate(RZGate(), 0)

    for _ in range(100):
        utry = UnitaryMatrix.random(1)
        circuit.instantiate(
            utry,
            method='minimization',
            minimizer=ScipyMinimizer(),
            cost_fn_gen=HilbertSchmidtCostGenerator(),
        )
        assert circuit.get_unitary().get_distance_from(utry, 1) < 1e-12
