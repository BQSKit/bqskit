"""
This test module verifies all circuit properties.

The Circuit class defines multiple properties, but also inherits many from the
Unitary base class.

This test is broken down into multiple parts. First, a few simple known circuits
have their properties tested. Then, each property is tested in depth
individually.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import ConstantUnitaryGate
from bqskit.ir.gates import CSUMGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import TdgGate
from bqskit.ir.gates import TGate
from bqskit.ir.gates import U3Gate
from bqskit.ir.gates import XGate
from bqskit.ir.gates import ZGate
from bqskit.qis.graph import CouplingGraph
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_numeric
from bqskit.utils.typing import is_valid_radixes


class TestSimpleCircuit:
    """This set of tests will ensure that all circuit properties are correct for
    a simple circuit."""

    def test_num_params(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.num_params == 0

    def test_radixes(self, simple_circuit: Circuit) -> None:
        assert len(simple_circuit.radixes) == simple_circuit.num_qudits
        assert isinstance(simple_circuit.radixes, tuple)
        assert all(r == 2 for r in simple_circuit.radixes)

    def test_num_qudits(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.num_qudits == 2

    def test_dim(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.dim == 4

    def test_is_qubit_only(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.is_qubit_only()

    def test_is_qutrit_only(self, simple_circuit: Circuit) -> None:
        assert not simple_circuit.is_qutrit_only()

    def test_is_parameterized(self, simple_circuit: Circuit) -> None:
        assert not simple_circuit.is_parameterized()

    def test_is_constant(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.is_constant()

    def test_num_operations(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.num_operations == 4

    def test_num_cycles(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.num_cycles == 4

    def test_params(self, simple_circuit: Circuit) -> None:
        assert len(simple_circuit.params) == 0
        assert isinstance(simple_circuit.params, np.ndarray)

    def test_depth(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.depth == 4

    def test_parallelism(self, simple_circuit: Circuit) -> None:
        assert simple_circuit.parallelism == 1.5

    def test_coupling_graph(self, simple_circuit: Circuit) -> None:
        cgraph = simple_circuit.coupling_graph
        assert CouplingGraph.is_valid_coupling_graph(cgraph, 2)
        assert isinstance(cgraph, CouplingGraph)
        assert len(cgraph) == 1
        assert (0, 1) in cgraph

    def test_gate_set(self, simple_circuit: Circuit) -> None:
        gate_set = simple_circuit.gate_set
        assert isinstance(gate_set, set)
        assert len(gate_set) == 2
        assert XGate() in gate_set
        assert CNOTGate() in gate_set

    def test_active_qudits(self, simple_circuit: Circuit) -> None:
        qudits = simple_circuit.active_qudits
        assert len(qudits) == simple_circuit.num_qudits
        assert isinstance(qudits, list)
        assert all(x in qudits for x in range(simple_circuit.num_qudits))


class TestSwapCircuit:
    """This set of tests will ensure that all circuit properties are correct for
    a swap circuit."""

    def test_num_params(self, swap_circuit: Circuit) -> None:
        assert swap_circuit.num_params == 0

    def test_radixes(self, swap_circuit: Circuit) -> None:
        assert len(swap_circuit.radixes) == swap_circuit.num_qudits
        assert isinstance(swap_circuit.radixes, tuple)
        assert all(r == 2 for r in swap_circuit.radixes)

    def test_num_qudits(self, swap_circuit: Circuit) -> None:
        assert swap_circuit.num_qudits == 2

    def test_dim(self, swap_circuit: Circuit) -> None:
        assert swap_circuit.dim == 4

    def test_is_qubit_only(self, swap_circuit: Circuit) -> None:
        assert swap_circuit.is_qubit_only()

    def test_is_qutrit_only(self, swap_circuit: Circuit) -> None:
        assert not swap_circuit.is_qutrit_only()

    def test_is_parameterized(self, swap_circuit: Circuit) -> None:
        assert not swap_circuit.is_parameterized()

    def test_is_constant(self, swap_circuit: Circuit) -> None:
        assert swap_circuit.is_constant()

    def test_num_operations(self, swap_circuit: Circuit) -> None:
        assert swap_circuit.num_operations == 3

    def test_num_cycles(self, swap_circuit: Circuit) -> None:
        assert swap_circuit.num_cycles == 3

    def test_params(self, swap_circuit: Circuit) -> None:
        assert len(swap_circuit.params) == 0
        assert isinstance(swap_circuit.params, np.ndarray)

    def test_depth(self, swap_circuit: Circuit) -> None:
        assert swap_circuit.depth == 3

    def test_parallelism(self, swap_circuit: Circuit) -> None:
        assert swap_circuit.parallelism == 2

    def test_coupling_graph(self, swap_circuit: Circuit) -> None:
        cgraph = swap_circuit.coupling_graph
        assert CouplingGraph.is_valid_coupling_graph(cgraph, 2)
        assert isinstance(cgraph, CouplingGraph)
        assert len(cgraph) == 1
        assert (0, 1) in cgraph

    def test_gate_set(self, swap_circuit: Circuit) -> None:
        gate_set = swap_circuit.gate_set
        assert isinstance(gate_set, set)
        assert len(gate_set) == 1
        assert CNOTGate() in gate_set

    def test_active_qudits(self, swap_circuit: Circuit) -> None:
        qudits = swap_circuit.active_qudits
        assert len(qudits) == swap_circuit.num_qudits
        assert isinstance(qudits, list)
        assert all(x in qudits for x in range(swap_circuit.num_qudits))


class TestToffoliCircuit:
    """This set of tests will ensure that all circuit properties are correct for
    a toffoli circuit."""

    def test_num_params(self, toffoli_circuit: Circuit) -> None:
        assert toffoli_circuit.num_params == 0

    def test_radixes(self, toffoli_circuit: Circuit) -> None:
        assert len(toffoli_circuit.radixes) == toffoli_circuit.num_qudits
        assert isinstance(toffoli_circuit.radixes, tuple)
        assert all(r == 2 for r in toffoli_circuit.radixes)

    def test_num_qudits(self, toffoli_circuit: Circuit) -> None:
        assert toffoli_circuit.num_qudits == 3

    def test_dim(self, toffoli_circuit: Circuit) -> None:
        assert toffoli_circuit.dim == 8

    def test_is_qubit_only(self, toffoli_circuit: Circuit) -> None:
        assert toffoli_circuit.is_qubit_only()

    def test_is_qutrit_only(self, toffoli_circuit: Circuit) -> None:
        assert not toffoli_circuit.is_qutrit_only()

    def test_is_parameterized(self, toffoli_circuit: Circuit) -> None:
        assert not toffoli_circuit.is_parameterized()

    def test_is_constant(self, toffoli_circuit: Circuit) -> None:
        assert toffoli_circuit.is_constant()

    def test_num_operations(self, toffoli_circuit: Circuit) -> None:
        assert toffoli_circuit.num_operations == 15

    def test_num_cycles(self, toffoli_circuit: Circuit) -> None:
        assert toffoli_circuit.num_cycles == 11

    def test_params(self, toffoli_circuit: Circuit) -> None:
        assert len(toffoli_circuit.params) == 0
        assert isinstance(toffoli_circuit.params, np.ndarray)

    def test_depth(self, toffoli_circuit: Circuit) -> None:
        assert toffoli_circuit.depth == 11

    def test_parallelism(self, toffoli_circuit: Circuit) -> None:
        assert toffoli_circuit.parallelism == 21 / 11

    def test_coupling_graph(self, toffoli_circuit: Circuit) -> None:
        cgraph = toffoli_circuit.coupling_graph
        assert CouplingGraph.is_valid_coupling_graph(cgraph, 3)
        assert isinstance(cgraph, CouplingGraph)
        assert len(cgraph) == 3
        assert (0, 1) in cgraph
        assert (1, 2) in cgraph
        assert (0, 2) in cgraph

    def test_gate_set(self, toffoli_circuit: Circuit) -> None:
        gate_set = toffoli_circuit.gate_set
        assert isinstance(gate_set, set)
        assert len(gate_set) == 4
        assert CNOTGate() in gate_set
        assert HGate() in gate_set
        assert TdgGate() in gate_set
        assert TGate() in gate_set

    def test_active_qudits(self, toffoli_circuit: Circuit) -> None:
        qudits = toffoli_circuit.active_qudits
        assert len(qudits) == toffoli_circuit.num_qudits
        assert isinstance(qudits, list)
        assert all(x in qudits for x in range(toffoli_circuit.num_qudits))


class TestGetNumParams:
    """This tests `circuit.num_params`."""

    def test_type(self, r6_qudit_circuit: Circuit) -> None:
        assert isinstance(r6_qudit_circuit.num_params, int)

    def test_value(self, r6_qudit_circuit: Circuit) -> None:
        assert r6_qudit_circuit.num_params >= 0

    def test_empty(self) -> None:
        circuit = Circuit(1)
        assert circuit.num_params == 0
        circuit = Circuit(4)
        assert circuit.num_params == 0
        circuit = Circuit(4, [2, 3, 4, 5])
        assert circuit.num_params == 0

    def test_adding_gate(self) -> None:
        circuit = Circuit(1)
        assert circuit.num_params == 0
        circuit.append_gate(U3Gate(), [0])
        assert circuit.num_params == 3
        circuit.append_gate(U3Gate(), [0])
        assert circuit.num_params == 6
        circuit.append_gate(U3Gate(), [0])
        assert circuit.num_params == 9

    def test_inserting_gate(self) -> None:
        circuit = Circuit(1)
        assert circuit.num_params == 0
        circuit.insert_gate(0, U3Gate(), [0])
        assert circuit.num_params == 3
        circuit.insert_gate(0, U3Gate(), [0])
        assert circuit.num_params == 6
        circuit.insert_gate(0, U3Gate(), [0])
        assert circuit.num_params == 9

    def test_removing_gate(self) -> None:
        circuit = Circuit(1)
        circuit.append_gate(U3Gate(), [0])
        circuit.append_gate(U3Gate(), [0])
        circuit.append_gate(U3Gate(), [0])
        assert circuit.num_params == 9
        circuit.remove(U3Gate())
        assert circuit.num_params == 6
        circuit.remove(U3Gate())
        assert circuit.num_params == 3
        circuit.remove(U3Gate())
        assert circuit.num_params == 0

    def test_freezing_param(self) -> None:
        circuit = Circuit(1)
        circuit.append_gate(U3Gate(), [0])
        circuit.append_gate(U3Gate(), [0])
        circuit.append_gate(U3Gate(), [0])
        assert circuit.num_params == 9
        circuit.freeze_param(0)
        assert circuit.num_params == 8
        circuit.freeze_param(0)
        assert circuit.num_params == 7
        circuit.freeze_param(0)
        assert circuit.num_params == 6
        circuit.freeze_param(0)

    def test_r1(self, r3_qubit_circuit: Circuit) -> None:
        start = r3_qubit_circuit.num_params
        r3_qubit_circuit.append_gate(U3Gate(), [0])
        assert r3_qubit_circuit.num_params == start + 3
        r3_qubit_circuit.insert_gate(0, U3Gate(), [1])
        assert r3_qubit_circuit.num_params == start + 6
        r3_qubit_circuit.insert_gate(0, CNOTGate(), [0, 2])
        assert r3_qubit_circuit.num_params == start + 6
        r3_qubit_circuit.remove(U3Gate())
        assert r3_qubit_circuit.num_params == start + 3


class TestGetRadixes:
    """This tests `circuit.radixes`."""

    def test_type(self, r6_qudit_circuit: Circuit) -> None:
        assert isinstance(r6_qudit_circuit.radixes, tuple)
        assert all(is_integer(r) for r in r6_qudit_circuit.radixes)

    def test_value(self, r6_qudit_circuit: Circuit) -> None:
        assert is_valid_radixes(r6_qudit_circuit.radixes, 6)

    def test_empty(self) -> None:
        circuit = Circuit(1)
        assert len(circuit.radixes) == 1
        assert circuit.radixes[0] == 2
        circuit = Circuit(4)
        assert len(circuit.radixes) == 4
        assert circuit.radixes[0] == 2
        assert circuit.radixes[1] == 2
        assert circuit.radixes[2] == 2
        assert circuit.radixes[3] == 2
        circuit = Circuit(4, [2, 2, 3, 3])
        assert len(circuit.radixes) == 4
        assert circuit.radixes[0] == 2
        assert circuit.radixes[1] == 2
        assert circuit.radixes[2] == 3
        assert circuit.radixes[3] == 3


class TestGetSize:
    """This tests `circuit.num_qudits`."""

    def test_type(self, r6_qudit_circuit: Circuit) -> None:
        assert isinstance(r6_qudit_circuit.num_qudits, int)

    def test_value(self, r6_qudit_circuit: Circuit) -> None:
        assert r6_qudit_circuit.num_qudits == 6

    def test_empty(self) -> None:
        circuit = Circuit(1)
        assert circuit.num_qudits == 1
        circuit = Circuit(4)
        assert circuit.num_qudits == 4
        circuit = Circuit(4, [2, 2, 3, 3])
        assert circuit.num_qudits == 4


class TestGetDim:
    """This tests `circuit.dim`."""

    def test_type(self, r6_qudit_circuit: Circuit) -> None:
        assert isinstance(r6_qudit_circuit.dim, int)

    def test_value(self, r6_qudit_circuit: Circuit) -> None:
        assert r6_qudit_circuit.dim >= 64

    def test_empty(self) -> None:
        circuit = Circuit(1)
        assert circuit.dim == 2
        circuit = Circuit(4)
        assert circuit.dim == 16
        circuit = Circuit(4, [2, 2, 3, 3])
        assert circuit.dim == 36


class TestIsQubitOnly:
    """This tests `circuit.is_qubit_only`."""

    def test_type(self, r6_qudit_circuit: Circuit) -> None:
        assert isinstance(r6_qudit_circuit.is_qubit_only(), bool)

    def test_value(self, r6_qudit_circuit: Circuit) -> None:
        if r6_qudit_circuit.radixes.count(2) == 6:
            assert r6_qudit_circuit.is_qubit_only()
        else:
            assert not r6_qudit_circuit.is_qubit_only()

    def test_empty(self) -> None:
        circuit = Circuit(1)
        assert circuit.is_qubit_only()
        circuit = Circuit(4)
        assert circuit.is_qubit_only()
        circuit = Circuit(4, [2, 2, 3, 3])
        assert not circuit.is_qubit_only()


class TestIsQutritOnly:
    """This tests `circuit.is_qutrit_only`."""

    def test_type(self, r6_qudit_circuit: Circuit) -> None:
        assert isinstance(r6_qudit_circuit.is_qutrit_only(), bool)

    def test_value(self, r6_qudit_circuit: Circuit) -> None:
        if r6_qudit_circuit.radixes.count(3) == 6:
            assert r6_qudit_circuit.is_qutrit_only()
        else:
            assert not r6_qudit_circuit.is_qutrit_only()

    def test_empty(self) -> None:
        circuit = Circuit(1)
        assert not circuit.is_qutrit_only()
        circuit = Circuit(4)
        assert not circuit.is_qutrit_only()
        circuit = Circuit(4, [3, 3, 3, 3])
        assert circuit.is_qutrit_only()
        circuit = Circuit(4, [2, 2, 3, 3])
        assert not circuit.is_qutrit_only()


class TestIsParameterized:
    """This tests `circuit.is_parameterized`."""

    def test_type(self, r6_qudit_circuit: Circuit) -> None:
        assert isinstance(r6_qudit_circuit.is_parameterized(), bool)

    def test_value(self, r6_qudit_circuit: Circuit) -> None:
        assert (
            r6_qudit_circuit.is_parameterized()
            != r6_qudit_circuit.is_constant()
        )
        if any(g.is_parameterized() for g in r6_qudit_circuit.gate_set):
            assert r6_qudit_circuit.is_parameterized()
        else:
            assert not r6_qudit_circuit.is_parameterized()

    def test_empty(self) -> None:
        circuit = Circuit(1)
        assert not circuit.is_parameterized()
        circuit = Circuit(4)
        assert not circuit.is_parameterized()
        circuit = Circuit(4, [2, 2, 3, 3])
        assert not circuit.is_parameterized()


class TestIsConstant:
    """This tests `circuit.is_constant`."""

    def test_type(self, r6_qudit_circuit: Circuit) -> None:
        assert isinstance(r6_qudit_circuit.is_constant(), bool)

    def test_value(self, r6_qudit_circuit: Circuit) -> None:
        assert (
            r6_qudit_circuit.is_parameterized()
            != r6_qudit_circuit.is_constant()
        )
        if all(g.is_constant() for g in r6_qudit_circuit.gate_set):
            assert r6_qudit_circuit.is_constant()
        else:
            assert not r6_qudit_circuit.is_constant()

    def test_empty(self) -> None:
        circuit = Circuit(1)
        assert circuit.is_constant()
        circuit = Circuit(4)
        assert circuit.is_constant()
        circuit = Circuit(4, [2, 2, 3, 3])
        assert circuit.is_constant()


class TestGetNumOperations:
    """This tests `circuit.num_operations`."""

    def test_type(self, r6_qudit_circuit: Circuit) -> None:
        assert isinstance(r6_qudit_circuit.num_operations, int)

    def test_value(self, r6_qudit_circuit: Circuit) -> None:
        assert r6_qudit_circuit.num_operations >= 0

    def test_empty(self) -> None:
        circuit = Circuit(1)
        assert circuit.num_operations == 0
        circuit = Circuit(4)
        assert circuit.num_operations == 0
        circuit = Circuit(4, [2, 3, 4, 5])
        assert circuit.num_operations == 0

    def test_adding_gate(self) -> None:
        circuit = Circuit(1)
        assert circuit.num_operations == 0
        circuit.append_gate(U3Gate(), [0])
        assert circuit.num_operations == 1
        circuit.append_gate(U3Gate(), [0])
        assert circuit.num_operations == 2
        circuit.append_gate(U3Gate(), [0])
        assert circuit.num_operations == 3

    def test_inserting_gate(self) -> None:
        circuit = Circuit(1)
        assert circuit.num_operations == 0
        circuit.insert_gate(0, U3Gate(), [0])
        assert circuit.num_operations == 1
        circuit.insert_gate(0, U3Gate(), [0])
        assert circuit.num_operations == 2
        circuit.insert_gate(0, U3Gate(), [0])
        assert circuit.num_operations == 3

    def test_removing_gate(self) -> None:
        circuit = Circuit(1)
        circuit.append_gate(U3Gate(), [0])
        circuit.append_gate(U3Gate(), [0])
        circuit.append_gate(U3Gate(), [0])
        assert circuit.num_operations == 3
        circuit.remove(U3Gate())
        assert circuit.num_operations == 2
        circuit.remove(U3Gate())
        assert circuit.num_operations == 1
        circuit.remove(U3Gate())
        assert circuit.num_operations == 0

    def test_r1(self, r3_qubit_circuit: Circuit) -> None:
        assert r3_qubit_circuit.num_operations == 10
        r3_qubit_circuit.append_gate(U3Gate(), [0])
        assert r3_qubit_circuit.num_operations == 11
        r3_qubit_circuit.insert_gate(0, U3Gate(), [1])
        assert r3_qubit_circuit.num_operations == 12
        r3_qubit_circuit.insert_gate(0, CNOTGate(), [0, 2])
        assert r3_qubit_circuit.num_operations == 13
        r3_qubit_circuit.remove(U3Gate())
        assert r3_qubit_circuit.num_operations == 12
        r3_qubit_circuit.remove(CNOTGate())
        assert r3_qubit_circuit.num_operations == 11


class TestGetNumCycles:
    """This tests `circuit.num_cycles`."""

    def test_type(self, r6_qudit_circuit: Circuit) -> None:
        assert isinstance(r6_qudit_circuit.num_cycles, int)

    def test_value(self, r6_qudit_circuit: Circuit) -> None:
        assert r6_qudit_circuit.num_cycles >= 0

    def test_empty(self) -> None:
        circuit = Circuit(1)
        assert circuit.num_cycles == 0
        circuit = Circuit(4)
        assert circuit.num_cycles == 0
        circuit = Circuit(4, [2, 3, 4, 5])
        assert circuit.num_cycles == 0

    def test_adding_gate(self) -> None:
        circuit = Circuit(1)
        assert circuit.num_cycles == 0
        circuit.append_gate(U3Gate(), [0])
        assert circuit.num_cycles == 1
        circuit.append_gate(U3Gate(), [0])
        assert circuit.num_cycles == 2
        circuit.append_gate(U3Gate(), [0])
        assert circuit.num_cycles == 3

    def test_inserting_gate(self) -> None:
        circuit = Circuit(1)
        assert circuit.num_cycles == 0
        circuit.insert_gate(0, U3Gate(), [0])
        assert circuit.num_cycles == 1
        circuit.insert_gate(0, U3Gate(), [0])
        assert circuit.num_cycles == 2
        circuit.insert_gate(0, U3Gate(), [0])
        assert circuit.num_cycles == 3

    def test_removing_gate1(self) -> None:
        circuit = Circuit(1)
        circuit.append_gate(U3Gate(), [0])
        circuit.append_gate(U3Gate(), [0])
        circuit.append_gate(U3Gate(), [0])
        assert circuit.num_cycles == 3
        circuit.remove(U3Gate())
        assert circuit.num_cycles == 2
        circuit.remove(U3Gate())
        assert circuit.num_cycles == 1
        circuit.remove(U3Gate())
        assert circuit.num_cycles == 0

    def test_removing_gate2(self) -> None:
        circuit = Circuit(2)
        circuit.append_gate(U3Gate(), [0])
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(U3Gate(), [0])
        circuit.append_gate(U3Gate(), [1])
        assert circuit.num_cycles == 3
        circuit.remove(U3Gate())
        assert circuit.num_cycles == 2
        circuit.remove(CNOTGate())
        assert circuit.num_cycles == 1
        circuit.remove(U3Gate())
        assert circuit.num_cycles == 1
        circuit.remove(U3Gate())
        assert circuit.num_cycles == 0


class TestGetParams:
    """This tests `circuit.params`."""

    def test_type(self, r6_qudit_circuit: Circuit) -> None:
        params = r6_qudit_circuit.params
        assert isinstance(params, np.ndarray)
        assert all(is_numeric(param) for param in params)

    def test_count(self, r6_qudit_circuit: Circuit) -> None:
        num_params = r6_qudit_circuit.num_params
        params = r6_qudit_circuit.params
        assert len(params) == num_params

    def test_no_modify(self, r6_qudit_circuit: Circuit) -> None:
        params = r6_qudit_circuit.params
        if len(params) == 0:
            return
        params[0] = -params[0] + 1
        assert params[0] != r6_qudit_circuit.params[0]

    def test_empty(self) -> None:
        circuit = Circuit(1)
        assert len(circuit.params) == 0
        circuit = Circuit(4)
        assert len(circuit.params) == 0
        circuit = Circuit(4, [2, 3, 4, 5])
        assert len(circuit.params) == 0


class TestGetDepth:
    """This tests `circuit.depth`."""

    def test_type(self, r6_qudit_circuit: Circuit) -> None:
        assert isinstance(r6_qudit_circuit.depth, int)

    def test_value(self, r6_qudit_circuit: Circuit) -> None:
        assert r6_qudit_circuit.depth >= 0

    def test_empty(self) -> None:
        circuit = Circuit(1)
        assert circuit.depth == 0
        circuit = Circuit(4)
        assert circuit.depth == 0
        circuit = Circuit(4, [2, 3, 4, 5])
        assert circuit.depth == 0

    def test_adding_gate(self) -> None:
        circuit = Circuit(1)
        assert circuit.depth == 0
        circuit.append_gate(U3Gate(), [0])
        assert circuit.depth == 1
        circuit.append_gate(U3Gate(), [0])
        assert circuit.depth == 2
        circuit.append_gate(U3Gate(), [0])
        assert circuit.depth == 3

    def test_inserting_gate(self) -> None:
        circuit = Circuit(1)
        assert circuit.depth == 0
        circuit.insert_gate(0, U3Gate(), [0])
        assert circuit.depth == 1
        circuit.insert_gate(0, U3Gate(), [0])
        assert circuit.depth == 2
        circuit.insert_gate(0, U3Gate(), [0])
        assert circuit.depth == 3

    def test_removing_gate1(self) -> None:
        circuit = Circuit(1)
        circuit.append_gate(U3Gate(), [0])
        circuit.append_gate(U3Gate(), [0])
        circuit.append_gate(U3Gate(), [0])
        assert circuit.depth == 3
        circuit.remove(U3Gate())
        assert circuit.depth == 2
        circuit.remove(U3Gate())
        assert circuit.depth == 1
        circuit.remove(U3Gate())
        assert circuit.depth == 0

    def test_removing_gate2(self) -> None:
        circuit = Circuit(2)
        circuit.append_gate(U3Gate(), [0])
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(U3Gate(), [0])
        circuit.append_gate(U3Gate(), [1])
        assert circuit.depth == 3
        circuit.remove(U3Gate())
        assert circuit.depth == 2
        circuit.remove(CNOTGate())
        assert circuit.depth == 1
        circuit.remove(U3Gate())
        assert circuit.depth == 1
        circuit.remove(U3Gate())
        assert circuit.depth == 0

    def test_vs_cycles(self, r6_qudit_circuit: Circuit) -> None:
        assert (
            r6_qudit_circuit.depth
            <= r6_qudit_circuit.num_cycles
        )


class TestGetParallelism:
    """This tests `circuit.parallelism`."""

    def test_type(self, r6_qudit_circuit: Circuit) -> None:
        assert isinstance(r6_qudit_circuit.parallelism, float)

    def test_value(self, r6_qudit_circuit: Circuit) -> None:
        assert r6_qudit_circuit.parallelism > 0

    def test_empty(self) -> None:
        circuit = Circuit(1)
        assert circuit.parallelism == 0
        circuit = Circuit(4)
        assert circuit.parallelism == 0
        circuit = Circuit(4, [2, 3, 4, 5])
        assert circuit.parallelism == 0

    def test_adding_gate(self) -> None:
        circuit = Circuit(1)
        assert circuit.parallelism == 0
        circuit.append_gate(U3Gate(), [0])
        assert circuit.parallelism == 1
        circuit.append_gate(U3Gate(), [0])
        assert circuit.parallelism == 1
        circuit.append_gate(U3Gate(), [0])
        assert circuit.parallelism == 1

    def test_adding_gate_2(self) -> None:
        circuit = Circuit(2)
        assert circuit.parallelism == 0
        circuit.append_gate(U3Gate(), [0])
        assert circuit.parallelism == 1
        circuit.append_gate(U3Gate(), [1])
        assert circuit.parallelism == 2
        circuit.append_gate(U3Gate(), [0])
        assert circuit.parallelism == 1.5
        circuit.append_gate(U3Gate(), [1])
        assert circuit.parallelism == 2
        circuit.append_gate(U3Gate(), [0])
        assert circuit.parallelism - 5 / 3 < 1e-12
        circuit.append_gate(U3Gate(), [1])
        assert circuit.parallelism == 2

    def test_adding_gate_3(self) -> None:
        circuit = Circuit(2)
        assert circuit.parallelism == 0
        circuit.append_gate(U3Gate(), [0])
        assert circuit.parallelism == 1
        circuit.append_gate(U3Gate(), [1])
        assert circuit.parallelism == 2
        circuit.append_gate(CNOTGate(), [0, 1])
        assert circuit.parallelism == 2


class TestGetCouplingGraph:
    """This tests `circuit.coupling_graph`."""

    def test_type(self, r6_qudit_circuit: Circuit) -> None:
        assert CouplingGraph.is_valid_coupling_graph(
            r6_qudit_circuit.coupling_graph, 6,
        )

    def test_empty(self) -> None:
        circuit = Circuit(4)
        assert len(circuit.coupling_graph) == 0
        assert isinstance(circuit.coupling_graph, CouplingGraph)

    def test_single_qubit_1(self) -> None:
        circuit = Circuit(1)
        assert len(circuit.coupling_graph) == 0
        circuit.append_gate(U3Gate(), [0])
        assert len(circuit.coupling_graph) == 0
        circuit.append_gate(U3Gate(), [0])
        assert len(circuit.coupling_graph) == 0
        circuit.append_gate(U3Gate(), [0])
        assert len(circuit.coupling_graph) == 0

    def test_single_qubit_2(self) -> None:
        circuit = Circuit(4)
        assert len(circuit.coupling_graph) == 0
        for i in range(4):
            circuit.append_gate(U3Gate(), [i])
        assert len(circuit.coupling_graph) == 0
        for j in range(4):
            for i in range(4):
                circuit.append_gate(U3Gate(), [i])
        assert len(circuit.coupling_graph) == 0

    def test_two_qubit_1(self) -> None:
        circuit = Circuit(2)
        assert len(circuit.coupling_graph) == 0

        circuit.append_gate(CNOTGate(), [0, 1])
        cgraph = circuit.coupling_graph
        assert len(cgraph) == 1
        assert (0, 1) in cgraph

        circuit.append_gate(CNOTGate(), [1, 0])
        cgraph = circuit.coupling_graph
        assert len(cgraph) == 1
        assert (0, 1) in cgraph

        circuit.remove(CNOTGate())
        circuit.remove(CNOTGate())
        assert len(circuit.coupling_graph) == 0

    def test_two_qubit_2(self) -> None:
        circuit = Circuit(4)
        assert len(circuit.coupling_graph) == 0

        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        cgraph = circuit.coupling_graph
        assert len(cgraph) == 3
        assert (0, 1) in cgraph
        assert (1, 2) in cgraph
        assert (2, 3) in cgraph

        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [0, 1])
        cgraph = circuit.coupling_graph
        assert len(cgraph) == 3
        assert (0, 1) in cgraph
        assert (1, 2) in cgraph
        assert (2, 3) in cgraph

        circuit.append_gate(CNOTGate(), [0, 2])
        circuit.append_gate(CNOTGate(), [3, 0])
        cgraph = circuit.coupling_graph
        assert len(cgraph) == 5
        assert (0, 1) in cgraph
        assert (1, 2) in cgraph
        assert (2, 3) in cgraph
        assert (0, 2) in cgraph
        assert (0, 3) in cgraph

    def test_multi_qubit_1(self, gen_random_utry_np: Any) -> None:
        circuit = Circuit(6)
        assert len(circuit.coupling_graph) == 0

        three_qubit_gate = ConstantUnitaryGate(gen_random_utry_np(8))
        circuit.append_gate(three_qubit_gate, [0, 1, 2])
        cgraph = circuit.coupling_graph
        assert len(cgraph) == 3
        assert (0, 1) in cgraph
        assert (1, 2) in cgraph
        assert (0, 2) in cgraph

        circuit.append_gate(three_qubit_gate, [0, 1, 2])
        cgraph = circuit.coupling_graph
        assert len(cgraph) == 3
        assert (0, 1) in cgraph
        assert (1, 2) in cgraph
        assert (0, 2) in cgraph

        circuit.append_gate(three_qubit_gate, [1, 2, 3])
        cgraph = circuit.coupling_graph
        assert len(cgraph) == 5
        assert (0, 1) in cgraph
        assert (1, 2) in cgraph
        assert (0, 2) in cgraph
        assert (1, 3) in cgraph
        assert (2, 3) in cgraph

        circuit.append_gate(three_qubit_gate, [3, 4, 5])
        cgraph = circuit.coupling_graph
        assert len(cgraph) == 8
        assert (0, 1) in cgraph
        assert (1, 2) in cgraph
        assert (0, 2) in cgraph
        assert (1, 3) in cgraph
        assert (2, 3) in cgraph
        assert (3, 4) in cgraph
        assert (3, 5) in cgraph
        assert (4, 5) in cgraph

    def test_multi_qudit_2(self, gen_random_utry_np: Any) -> None:
        circuit = Circuit(6, [2, 2, 2, 3, 3, 3])
        assert len(circuit.coupling_graph) == 0

        three_qubit_gate = ConstantUnitaryGate(
            gen_random_utry_np(12), [2, 2, 3],
        )
        circuit.append_gate(three_qubit_gate, [0, 1, 3])
        cgraph = circuit.coupling_graph
        assert len(cgraph) == 3
        assert (0, 1) in cgraph
        assert (1, 3) in cgraph
        assert (0, 3) in cgraph

        circuit.append_gate(CNOTGate(), [1, 2])
        cgraph = circuit.coupling_graph
        assert len(cgraph) == 4
        assert (0, 1) in cgraph
        assert (1, 3) in cgraph
        assert (0, 3) in cgraph
        assert (1, 2) in cgraph

        circuit.append_gate(CSUMGate(), [4, 5])
        cgraph = circuit.coupling_graph
        assert len(cgraph) == 5
        assert (0, 1) in cgraph
        assert (1, 3) in cgraph
        assert (0, 3) in cgraph
        assert (1, 2) in cgraph
        assert (4, 5) in cgraph


class TestGetGateSet:
    """This tests `circuit.gate_set`."""

    def test_type(self, r6_qudit_circuit: Circuit) -> None:
        assert isinstance(r6_qudit_circuit.gate_set, set)
        assert all(
            isinstance(gate, Gate)
            for gate in r6_qudit_circuit.gate_set
        )

    def test_empty(self) -> None:
        circuit = Circuit(4)
        assert len(circuit.gate_set) == 0
        assert isinstance(circuit.gate_set, set)

    def test_adding_gate(self) -> None:
        circuit = Circuit(1)
        assert len(circuit.gate_set) == 0
        circuit.append_gate(U3Gate(), [0])
        assert len(circuit.gate_set) == 1
        assert U3Gate() in circuit.gate_set
        circuit.append_gate(XGate(), [0])
        assert len(circuit.gate_set) == 2
        assert U3Gate() in circuit.gate_set
        assert XGate() in circuit.gate_set
        circuit.append_gate(ZGate(), [0])
        assert len(circuit.gate_set) == 3
        assert U3Gate() in circuit.gate_set
        assert XGate() in circuit.gate_set
        assert ZGate() in circuit.gate_set
        circuit.append_gate(TGate(), [0])
        assert len(circuit.gate_set) == 4
        assert U3Gate() in circuit.gate_set
        assert XGate() in circuit.gate_set
        assert ZGate() in circuit.gate_set
        assert TGate() in circuit.gate_set

    def test_removing_gate(self) -> None:
        circuit = Circuit(1)
        assert len(circuit.gate_set) == 0
        circuit.append_gate(U3Gate(), [0])
        circuit.append_gate(XGate(), [0])
        circuit.append_gate(ZGate(), [0])
        circuit.append_gate(TGate(), [0])
        assert len(circuit.gate_set) == 4
        assert U3Gate() in circuit.gate_set
        assert XGate() in circuit.gate_set
        assert ZGate() in circuit.gate_set
        assert TGate() in circuit.gate_set
        circuit.remove(TGate())
        assert len(circuit.gate_set) == 3
        assert U3Gate() in circuit.gate_set
        assert XGate() in circuit.gate_set
        assert ZGate() in circuit.gate_set
        circuit.remove(XGate())
        assert len(circuit.gate_set) == 2
        assert U3Gate() in circuit.gate_set
        assert ZGate() in circuit.gate_set
        circuit.remove(ZGate())
        assert len(circuit.gate_set) == 1
        assert U3Gate() in circuit.gate_set
        circuit.remove(U3Gate())
        assert len(circuit.gate_set) == 0

    def test_qudit(self) -> None:
        circuit = Circuit(3, [2, 3, 3])
        assert len(circuit.gate_set) == 0
        circuit.append_gate(U3Gate(), [0])
        assert len(circuit.gate_set) == 1
        assert U3Gate() in circuit.gate_set
        circuit.append_gate(XGate(), [0])
        assert len(circuit.gate_set) == 2
        assert U3Gate() in circuit.gate_set
        assert XGate() in circuit.gate_set
        circuit.append_gate(ZGate(), [0])
        assert len(circuit.gate_set) == 3
        assert U3Gate() in circuit.gate_set
        assert XGate() in circuit.gate_set
        assert ZGate() in circuit.gate_set
        circuit.append_gate(TGate(), [0])
        assert len(circuit.gate_set) == 4
        assert U3Gate() in circuit.gate_set
        assert XGate() in circuit.gate_set
        assert ZGate() in circuit.gate_set
        assert TGate() in circuit.gate_set
        circuit.append_gate(CSUMGate(), [1, 2])
        assert len(circuit.gate_set) == 5
        assert U3Gate() in circuit.gate_set
        assert XGate() in circuit.gate_set
        assert ZGate() in circuit.gate_set
        assert TGate() in circuit.gate_set
        assert CSUMGate() in circuit.gate_set


class TestIsDifferentiable:
    """This tests `circuit.is_differentiable`."""

    def test_type(self, r6_qudit_circuit: Circuit) -> None:
        assert isinstance(r6_qudit_circuit.is_differentiable(), bool)

    def test_value(self, gate: Gate) -> None:
        circuit = Circuit(gate.num_qudits, gate.radixes)
        assert circuit.is_differentiable()

        circuit.append_gate(gate, list(range(gate.num_qudits)))
        if isinstance(gate, DifferentiableUnitary):
            assert circuit.is_differentiable()
        else:
            assert not circuit.is_differentiable()

    @pytest.mark.parametrize(
        'circuit', [
            Circuit(1),
            Circuit(4),
            Circuit(4, [2, 3, 4, 5]),
        ],
    )
    def test_empty(self, circuit: Circuit) -> None:
        assert circuit.is_differentiable()
