"""
This test module verifies all circuit properties.

The Circuit class defines multiple properties, but also inherits many
from the Unitary base class.

Unitary base class properties:
    get_num_params(self) -> int
    get_radixes(self) -> tuple[int, ...]
    get_size(self) -> int
    get_dim(self) -> int
    is_qubit_only(self) -> bool
    is_qutrit_only(self) -> bool
    is_parameterized(self) -> bool
    is_constant(self) -> bool

Circuit class properties:
    get_num_operations(self) -> int
    get_num_cycles(self) -> int
    get_params(self) -> list[float]
    get_depth(self) -> int
    get_parallelism(self) -> float
    get_coupling_graph(self) -> set[tuple[int, int]]
    get_gate_set(self) -> set[Gate]

This test is broken down into multiple parts. First, a few simple known
circuits have their properties tested. Then, each property is tested
in depth.
"""

from __future__ import annotations
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.x import XGate

from bqskit.utils.typing import is_valid_coupling_graph
from bqskit.ir.circuit import Circuit


class TestSimpleCircuit:
    """
    This set of tests will ensure that all circuit properties
    are correct for a simple circuit.
    """
    def test_get_num_params(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.get_num_params() == 0

    def test_get_radixes(self, simple_circuit: Circuit) -> None:
        assert len(simple_circuit.get_radixes()) == simple_circuit.get_size()
        assert isinstance(simple_circuit.get_radixes(), tuple)
        assert all(r == 2 for r in simple_circuit.get_radixes())

    def test_get_size(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.get_size() == 2

    def test_get_dim(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.get_dim() == 4

    def test_is_qubit_only(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.is_qubit_only()

    def test_is_qutrit_only(self, simple_circuit: Circuit) -> None:
        assert not simple_circuit.is_qutrit_only()

    def test_is_parameterized(self, simple_circuit: Circuit) -> None:
        assert not simple_circuit.is_parameterized()

    def test_is_constant(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.is_constant()
    
    def test_get_num_operations(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.get_num_operations() == 4
    
    def test_get_num_cycles(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.get_num_cycles() == 4

    def test_get_params(self, simple_circuit: Circuit) -> None:
        assert len(simple_circuit.get_params()) == 0
        assert isinstance(simple_circuit.get_params(), list)

    def test_get_depth(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.get_depth() == 4

    def test_get_parallelism(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.get_parallelism() == 1

    def test_get_coupling_graph(self, simple_circuit: Circuit) -> None:
        cgraph = simple_circuit.get_coupling_graph()
        assert isinstance(cgraph, set)
        assert is_valid_coupling_graph(cgraph, 2)
        assert len(cgraph) == 1
        assert (0, 1) in cgraph
    
    def test_get_gate_set(self, simple_circuit: Circuit) -> None:
        gate_set = simple_circuit.get_gate_set()
        assert isinstance(gate_set, set)
        assert len(gate_set) == 2
        assert XGate() in gate_set
        assert CNOTGate() in gate_set
