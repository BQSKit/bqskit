from __future__ import annotations

from bqskit import compile
from bqskit.ir import Circuit
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import U3Gate
from bqskit.ir.opt.cost import HilbertSchmidtCostGenerator
from bqskit.ir.opt.minimizers import LBFGSMinimizer
from bqskit.qis.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary import UnitaryMatrix


def test_state_prep() -> None:
    state = StateVector.random(3)
    out_circuit = compile(state)
    out_state = out_circuit.get_statevector(StateVector.zero(3))
    assert out_state.get_distance_from(state) < 1e-8


def test_state_inst_2() -> None:
    state = StateVector.random(2)
    circuit = Circuit(2)
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(U3Gate(), 1)
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(U3Gate(), 1)
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(U3Gate(), 1)
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(U3Gate(), 1)
    circuit.instantiate(
        state,
        minimizer=LBFGSMinimizer(),
        cost_fn_gen=HilbertSchmidtCostGenerator(),
    )
    out_state = circuit.get_statevector(StateVector.zero(2))
    assert out_state.get_distance_from(state) < 1e-8


def test_state_inst_3() -> None:
    state = StateVector.random(3)
    circuit = Circuit(3)
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(U3Gate(), 1)
    circuit.append_gate(U3Gate(), 2)
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(CXGate(), (1, 2))
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(U3Gate(), 1)
    circuit.append_gate(U3Gate(), 2)
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(CXGate(), (1, 2))
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(U3Gate(), 1)
    circuit.append_gate(U3Gate(), 2)
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(CXGate(), (1, 2))
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(U3Gate(), 1)
    circuit.append_gate(U3Gate(), 2)
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(CXGate(), (1, 2))
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(U3Gate(), 1)
    circuit.append_gate(U3Gate(), 2)
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(CXGate(), (1, 2))
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(U3Gate(), 1)
    circuit.append_gate(U3Gate(), 2)
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(CXGate(), (1, 2))
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(U3Gate(), 1)
    circuit.append_gate(U3Gate(), 2)
    circuit.instantiate(
        state,
        minimizer=LBFGSMinimizer(),
        cost_fn_gen=HilbertSchmidtCostGenerator(),
    )
    out_state = circuit.get_statevector(StateVector.zero(3))
    assert out_state.get_distance_from(state) < 1e-8


def test_state_map() -> None:
    num_qudits = 3
    utry = UnitaryMatrix.random(num_qudits)
    state_map = {}
    for i in range(2):
        in_state = StateVector.random(num_qudits)
        out_state = utry.get_statevector(in_state)
        state_map[in_state] = out_state
    system = StateSystem(state_map)
    out_circuit = compile(system)
    print(out_circuit.gate_counts)

    for in_state, out_state in state_map.items():
        test_out_state = out_circuit.get_statevector(in_state)
        assert out_state.get_distance_from(test_out_state) < 1e-5
