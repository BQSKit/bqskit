"""This module tests the CircuitGate class."""
from __future__ import annotations

import pickle

from hypothesis import given
from hypothesis import HealthCheck
from hypothesis import settings

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.utils.test.strategies import circuits


class TestPickle:

    @settings(suppress_health_check=(HealthCheck.too_slow,))
    @given(circuits([2, 2], max_gates=3), circuits([2, 2], max_gates=3))
    def test_pickle_individual(self, c1: Circuit, c2: Circuit) -> None:
        gate1 = CircuitGate(c1)
        gate2 = CircuitGate(c2)
        utry1 = gate1.get_unitary()
        utry2 = gate2.get_unitary()
        pickled_utry1 = pickle.loads(pickle.dumps(gate1)).get_unitary()
        pickled_utry2 = pickle.loads(pickle.dumps(gate2)).get_unitary()
        assert utry1 == pickled_utry1
        assert utry2 == pickled_utry2

    @settings(suppress_health_check=(HealthCheck.too_slow,))
    @given(circuits([2, 2], max_gates=3), circuits([2, 2], max_gates=3))
    def test_pickle_list(self, c1: Circuit, c2: Circuit) -> None:
        gate1 = CircuitGate(c1)
        gate2 = CircuitGate(c2)
        utry1 = gate1.get_unitary()
        utry2 = gate2.get_unitary()
        pickled = pickle.loads(pickle.dumps([gate1, gate2]))
        assert utry1 == pickled[0].get_unitary()
        assert utry2 == pickled[1].get_unitary()

    @settings(suppress_health_check=(HealthCheck.too_slow,))
    @given(circuits([2, 2], max_gates=5), circuits([2, 2], max_gates=5))
    def test_pickle_circuit(self, c1: Circuit, c2: Circuit) -> None:
        gate1 = CircuitGate(c1)
        gate2 = CircuitGate(c2)
        circuit = Circuit(2)
        circuit.append_gate(gate1, [0, 1])
        circuit.append_gate(gate2, [0, 1])
        utry = circuit.get_unitary()
        pickled = pickle.loads(pickle.dumps(circuit))
        assert utry == pickled.get_unitary()
