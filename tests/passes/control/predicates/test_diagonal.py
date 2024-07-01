from __future__ import annotations

from hypothesis import given
from hypothesis.strategies import integers

import numpy as np

from itertools import combinations

from random import choices

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import SXGate
from bqskit.passes.control.predicates.diagonal import DiagonalPredicate


def phase_gadget() -> Circuit:
    gadget = Circuit(2)
    gadget.append_gate(CNOTGate(), (0, 1))
    gadget.append_gate(RZGate(), (1), [np.random.normal()])
    gadget.append_gate(CNOTGate(), (0, 1))
    return gadget

@given(integers(2, 6), integers(0, 10))
def test_diagonal_predicate(num_qudits: int, num_gadgets: int) -> None:
    circuit = Circuit(num_qudits)
    all_locations = list(combinations(range(num_qudits), r=2))
    locations = choices(all_locations, k=num_gadgets)
    for location in locations:
        circuit.append_circuit(phase_gadget(), location)
    data = PassData(circuit)
    pred = DiagonalPredicate(1e-5)
    assert pred.get_truth_value(circuit, data) == True

    circuit.append_gate(HGate(), (0))
    data = PassData(circuit)
    assert pred.get_truth_value(circuit, data) == False

@given(integers(1, 10))
def test_single_qubit_diagonal_predicate(exponent: int) -> None:
    angle = 10 ** - exponent
    circuit = Circuit(1)
    circuit.append_gate(RZGate(), (0), [angle])
    circuit.append_gate(SXGate(), (0))
    circuit.append_gate(RZGate(), (0), [np.random.normal()])
    circuit.append_gate(SXGate(), (0))
    circuit.append_gate(RZGate(), (0), [angle])

    pred = DiagonalPredicate(1e-5)
    data = PassData(circuit)
    # This is true by the small angle approximation
    pred.get_truth_value(circuit, data) == (angle < 1e-5)