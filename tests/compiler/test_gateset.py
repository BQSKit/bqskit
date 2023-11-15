"""This file tests the GateSet class."""
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis.strategies import builds
from hypothesis.strategies import sets

from bqskit.compiler.gateset import GateSet
from bqskit.ir.gate import Gate
from bqskit.ir.gates.constant.ccx import ToffoliGate
from bqskit.ir.gates.constant.csum import CSUMGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.cz import CZGate
from bqskit.ir.gates.constant.sx import SXGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.constant.y import YGate
from bqskit.ir.gates.parameterized.cphase import ArbitraryCPhaseGate
from bqskit.ir.gates.parameterized.pauli import PauliGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.parameterized.u8 import U8Gate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.passes.search.generators.fourparam import FourParamGenerator
from bqskit.passes.search.generators.simple import SimpleLayerGenerator
from bqskit.passes.search.generators.wide import WideLayerGenerator
from bqskit.utils.test.strategies import gates


def test_gate_set_init() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert len(gate_set) == 2
    assert CNOTGate() in gate_set
    assert U3Gate() in gate_set
    assert XGate() not in gate_set
    assert YGate() not in gate_set


def test_gate_set_empty() -> None:
    gate_set = GateSet({})
    assert len(gate_set) == 0
    assert gate_set.radix_set == set()
    with pytest.raises(RuntimeError):
        gate_set.build_layer_generator()

    with pytest.raises(RuntimeError):
        gate_set.build_mq_layer_generator()

    with pytest.raises(RuntimeError):
        gate_set.get_general_sq_gate()


def test_gate_set_build_layer_generator_simple() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    layergen = gate_set.build_layer_generator()
    assert isinstance(layergen, SimpleLayerGenerator)
    assert layergen.two_qudit_gate == CNOTGate()
    assert layergen.initial_layer_gate == U3Gate()
    assert layergen.single_qudit_gate_1 == U3Gate()
    assert layergen.single_qudit_gate_2 == U3Gate()


def test_gate_set_build_layer_generator_2q_basic() -> None:
    gate_set = GateSet({CZGate(), PauliGate(1), XGate()})
    layergen = gate_set.build_layer_generator()
    assert isinstance(layergen, SimpleLayerGenerator)
    assert layergen.two_qudit_gate == CZGate()
    assert layergen.initial_layer_gate == PauliGate(1)
    assert layergen.single_qudit_gate_1 == PauliGate(1)
    assert layergen.single_qudit_gate_2 == PauliGate(1)


def test_gate_set_build_layer_generator_3q_basic() -> None:
    gate_set = GateSet({ToffoliGate(), CZGate(), U3Gate()})
    layergen = gate_set.build_layer_generator()
    assert isinstance(layergen, WideLayerGenerator)
    assert ToffoliGate() in layergen.multi_qudit_gates
    assert CZGate() in layergen.multi_qudit_gates
    assert layergen.single_qudit_gate == U3Gate()


def test_gate_set_build_layer_generator_qutrit() -> None:
    gate_set = GateSet.default_gate_set(3)
    layergen = gate_set.build_layer_generator()
    assert isinstance(layergen, SimpleLayerGenerator)
    assert layergen.two_qudit_gate == CSUMGate(3)
    assert layergen.initial_layer_gate == VariableUnitaryGate(1, [3])
    assert layergen.single_qudit_gate_1 == VariableUnitaryGate(1, [3])
    assert layergen.single_qudit_gate_2 == VariableUnitaryGate(1, [3])

# TODO: Add tests for build_layer_generator with hybrid qudits


def test_gate_set_build_mq_layer_generator_simple() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    layergen = gate_set.build_mq_layer_generator()
    assert isinstance(layergen, FourParamGenerator)


def test_gate_set_build_mq_layer_generator_2q_basic() -> None:
    gate_set = GateSet({CZGate(), RZGate(), SXGate()})
    layergen = gate_set.build_mq_layer_generator()
    assert isinstance(layergen, SimpleLayerGenerator)
    assert layergen.two_qudit_gate == CZGate()
    assert layergen.initial_layer_gate == U3Gate()
    assert layergen.single_qudit_gate_1 == U3Gate()
    assert layergen.single_qudit_gate_2 == U3Gate()


def test_gate_set_build_mq_layer_generator_qutrit() -> None:
    gate_set = GateSet.default_gate_set(3)
    layergen = gate_set.build_mq_layer_generator()
    assert isinstance(layergen, SimpleLayerGenerator)
    assert layergen.two_qudit_gate == CSUMGate(3)
    assert layergen.initial_layer_gate == VariableUnitaryGate(1, [3])
    assert layergen.single_qudit_gate_1 == VariableUnitaryGate(1, [3])
    assert layergen.single_qudit_gate_2 == VariableUnitaryGate(1, [3])


def test_gate_set_get_general_sq_gate() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set.get_general_sq_gate() == U3Gate()

    gate_set = GateSet({CNOTGate(), SXGate(), RZGate()})
    assert gate_set.get_general_sq_gate() == U3Gate()

    gate_set = GateSet({ArbitraryCPhaseGate([3, 3]), U8Gate()})
    assert gate_set.get_general_sq_gate() == U8Gate()

    gate_set = GateSet({
        ArbitraryCPhaseGate(
            [3, 3],
        ), VariableUnitaryGate(1, [3]),
    })
    assert gate_set.get_general_sq_gate() == VariableUnitaryGate(1, [3])


def test_gate_set_default_gate_set() -> None:
    gate_set = GateSet.default_gate_set()
    assert len(gate_set) == 2
    assert CNOTGate() in gate_set
    assert U3Gate() in gate_set


def test_gate_set_default_gate_set_qutrits() -> None:
    gate_set = GateSet.default_gate_set(3)
    assert len(gate_set) == 2
    assert VariableUnitaryGate(1, [3]) in gate_set
    assert CSUMGate(3) in gate_set


def test_gate_set_default_gate_set_hybrid() -> None:
    gate_set = GateSet.default_gate_set([2, 3])
    assert len(gate_set) == 5
    assert VariableUnitaryGate(1, [2]) in gate_set
    assert VariableUnitaryGate(1, [3]) in gate_set
    assert ArbitraryCPhaseGate([2, 2]) in gate_set
    assert ArbitraryCPhaseGate([2, 3]) in gate_set
    assert ArbitraryCPhaseGate([3, 3]) in gate_set


def test_gate_set_single_qudit_gates() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert len(gate_set.single_qudit_gates) == 1
    assert U3Gate() in gate_set.single_qudit_gates
    assert CNOTGate() not in gate_set.single_qudit_gates


def test_gate_set_two_qudit_gates() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert len(gate_set.two_qudit_gates) == 1
    assert CNOTGate() in gate_set.two_qudit_gates
    assert U3Gate() not in gate_set.two_qudit_gates


def test_gate_set_many_qudit_gates() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate(), ToffoliGate()})
    assert len(gate_set.many_qudit_gates) == 1
    assert CNOTGate() not in gate_set.many_qudit_gates
    assert U3Gate() not in gate_set.many_qudit_gates
    assert ToffoliGate() in gate_set.many_qudit_gates


def test_gate_set_multi_qudit_gates() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate(), ToffoliGate()})
    assert len(gate_set.many_qudit_gates) == 1
    assert CNOTGate() not in gate_set.many_qudit_gates
    assert U3Gate() not in gate_set.many_qudit_gates
    assert ToffoliGate() in gate_set.many_qudit_gates


@given(builds(GateSet, sets(gates())))
def test_gate_set_properties(gate_set: GateSet) -> None:
    assert all(g.num_qudits == 1 for g in gate_set.single_qudit_gates)
    assert all(g.num_qudits == 2 for g in gate_set.two_qudit_gates)
    assert all(g.num_qudits > 2 for g in gate_set.many_qudit_gates)
    assert all(g.num_qudits >= 2 for g in gate_set.multi_qudit_gates)


def test_gate_set_union() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set.union({CNOTGate()}) == {CNOTGate(), U3Gate()}
    assert gate_set.union({U3Gate()}) == {CNOTGate(), U3Gate()}
    assert gate_set.union({CNOTGate(), U3Gate()}) == {CNOTGate(), U3Gate()}
    assert len(GateSet({U3Gate()}).union(gate_set)) == 2
    assert len(GateSet({CNOTGate()}).union(gate_set)) == 2


def test_gate_set_intesection() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set.intersection({CNOTGate()}) == {CNOTGate()}
    assert gate_set.intersection({U3Gate()}) == {U3Gate()}
    assert gate_set.intersection({CNOTGate(), U3Gate()}) == {
        CNOTGate(), U3Gate(),
    }
    assert len(GateSet({U3Gate()}).intersection(gate_set)) == 1


def test_gate_set_difference() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set.difference({CNOTGate()}) == {U3Gate()}
    assert gate_set.difference({U3Gate()}) == {CNOTGate()}
    assert gate_set.difference({CNOTGate(), U3Gate()}) == set()
    assert len(GateSet({U3Gate()}).difference(gate_set)) == 0


def test_gate_set_symmetric_difference() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set.symmetric_difference({CNOTGate()}) == {U3Gate()}
    assert gate_set.symmetric_difference({U3Gate()}) == {CNOTGate()}
    assert gate_set.symmetric_difference({CNOTGate(), U3Gate()}) == set()
    assert len(GateSet({U3Gate()}).symmetric_difference(gate_set)) == 1

    gate_set2 = GateSet({XGate(), YGate()})
    assert gate_set.symmetric_difference(
        gate_set2,
    ) == {CNOTGate(), U3Gate(), XGate(), YGate()}


def test_gate_set_issubset() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert not gate_set.issubset({CNOTGate()})
    assert not gate_set.issubset({U3Gate()})
    assert gate_set.issubset({CNOTGate(), U3Gate()})
    assert GateSet({U3Gate()}).issubset(gate_set)
    assert GateSet({CNOTGate()}).issubset(gate_set)


def test_gate_set_issuperset() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set.issuperset({CNOTGate()})
    assert gate_set.issuperset({U3Gate()})
    assert gate_set.issuperset({CNOTGate(), U3Gate()})
    assert not gate_set.issuperset({CNOTGate(), U3Gate(), ToffoliGate()})
    assert not GateSet({U3Gate()}).issuperset(gate_set)
    assert not GateSet({CNOTGate()}).issuperset(gate_set)


def test_gate_set_isdisjoint() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert not gate_set.isdisjoint({CNOTGate()})
    assert not gate_set.isdisjoint({U3Gate()})
    assert not gate_set.isdisjoint({CNOTGate(), U3Gate()})
    assert gate_set.isdisjoint({ToffoliGate()})


def test_gate_set_iter() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert len(list(gate_set)) == 2
    assert CNOTGate() in gate_set
    assert U3Gate() in gate_set
    for gate in gate_set:
        assert gate in gate_set


def test_gate_set_len() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert len(gate_set) == 2


def test_gate_set_contains() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert CNOTGate() in gate_set
    assert U3Gate() in gate_set
    assert ToffoliGate() not in gate_set


def test_gate_set_eq_other_gate_set() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set == gate_set
    assert gate_set == GateSet({CNOTGate(), U3Gate()})
    assert gate_set != GateSet({CNOTGate()})
    assert gate_set != GateSet({U3Gate()})
    assert gate_set != GateSet({CNOTGate(), U3Gate(), ToffoliGate()})
    assert gate_set != GateSet({ToffoliGate()})


def test_gate_set_eq_other_set() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set == {CNOTGate(), U3Gate()}
    assert gate_set != {CNOTGate()}
    assert gate_set != {U3Gate()}
    assert gate_set != {CNOTGate(), U3Gate(), ToffoliGate()}
    assert gate_set != {ToffoliGate()}


def test_gate_set_ne() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert not gate_set != gate_set
    assert not gate_set != GateSet({CNOTGate(), U3Gate()})
    assert gate_set != GateSet({CNOTGate()})
    assert gate_set != GateSet({U3Gate()})
    assert gate_set != GateSet({CNOTGate(), U3Gate(), ToffoliGate()})
    assert gate_set != GateSet({ToffoliGate()})


def test_gate_set_ne_other_set() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert not gate_set != {CNOTGate(), U3Gate()}
    assert gate_set != {CNOTGate()}
    assert gate_set != {U3Gate()}
    assert gate_set != {CNOTGate(), U3Gate(), ToffoliGate()}
    assert gate_set != {ToffoliGate()}


def test_gate_set_le() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set <= gate_set
    assert gate_set <= GateSet({CNOTGate(), U3Gate()})
    assert gate_set <= GateSet({CNOTGate(), U3Gate(), ToffoliGate()})
    assert not gate_set <= GateSet({CNOTGate()})
    assert not gate_set <= GateSet({U3Gate()})
    assert not gate_set <= GateSet({ToffoliGate()})


def test_gate_set_le_other_set() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set <= {CNOTGate(), U3Gate()}
    assert gate_set <= {CNOTGate(), U3Gate(), ToffoliGate()}
    assert not gate_set <= {CNOTGate()}
    assert not gate_set <= {U3Gate()}
    assert not gate_set <= {ToffoliGate()}


def test_gate_set_ge() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set >= gate_set
    assert gate_set >= GateSet({CNOTGate(), U3Gate()})
    assert gate_set >= GateSet({CNOTGate()})
    assert gate_set >= GateSet({U3Gate()})
    assert not gate_set >= GateSet({CNOTGate(), U3Gate(), ToffoliGate()})
    assert not gate_set >= GateSet({ToffoliGate()})


def test_gate_set_ge_other_set() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set >= {CNOTGate(), U3Gate()}
    assert gate_set >= {CNOTGate()}
    assert gate_set >= {U3Gate()}
    assert not gate_set >= {CNOTGate(), U3Gate(), ToffoliGate()}
    assert not gate_set >= {ToffoliGate()}


def test_gate_set_lt() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert not gate_set < gate_set
    assert not gate_set < GateSet({CNOTGate(), U3Gate()})
    assert gate_set < GateSet({CNOTGate(), U3Gate(), ToffoliGate()})
    assert not gate_set < GateSet({CNOTGate()})
    assert not gate_set < GateSet({U3Gate()})
    assert not gate_set < GateSet({ToffoliGate()})


def test_gate_set_lt_other_set() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert not gate_set < {CNOTGate(), U3Gate()}
    assert gate_set < {CNOTGate(), U3Gate(), ToffoliGate()}
    assert not gate_set < {CNOTGate()}
    assert not gate_set < {U3Gate()}
    assert not gate_set < {ToffoliGate()}


def test_gate_set_gt() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert not gate_set > gate_set
    assert not gate_set > GateSet({CNOTGate(), U3Gate()})
    assert gate_set > GateSet({CNOTGate()})
    assert gate_set > GateSet({U3Gate()})
    assert not gate_set > GateSet({CNOTGate(), U3Gate(), ToffoliGate()})
    assert not gate_set > GateSet({ToffoliGate()})


def test_gate_set_gt_other_set() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert not gate_set > {CNOTGate(), U3Gate()}
    assert gate_set > {CNOTGate()}
    assert gate_set > {U3Gate()}
    assert not gate_set > {CNOTGate(), U3Gate(), ToffoliGate()}
    assert not gate_set > {ToffoliGate()}


def test_gate_set_and() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set & gate_set == gate_set
    assert gate_set & GateSet({CNOTGate(), U3Gate()}) == gate_set
    assert gate_set & GateSet({CNOTGate()}) == GateSet({CNOTGate()})
    assert gate_set & GateSet({U3Gate()}) == GateSet({U3Gate()})
    assert gate_set & GateSet({CNOTGate(), U3Gate(), ToffoliGate()}) == gate_set
    assert gate_set & GateSet({ToffoliGate()}) == GateSet({})
    assert gate_set & GateSet({}) == GateSet({})


def test_gate_set_and_other_set() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set & {CNOTGate(), U3Gate()} == gate_set
    assert gate_set & {CNOTGate()} == GateSet({CNOTGate()})
    assert gate_set & {U3Gate()} == GateSet({U3Gate()})
    assert gate_set & {CNOTGate(), U3Gate(), ToffoliGate()} == gate_set
    assert gate_set & {ToffoliGate()} == GateSet({})
    assert gate_set & set() == GateSet({})


def test_gate_set_or() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set | gate_set == gate_set
    assert gate_set | GateSet({CNOTGate(), U3Gate()}) == gate_set
    assert gate_set | GateSet({CNOTGate()}) == gate_set
    assert gate_set | GateSet({U3Gate()}) == gate_set
    assert gate_set | GateSet({CNOTGate(), U3Gate(), ToffoliGate()}) == GateSet(
        {CNOTGate(), U3Gate(), ToffoliGate()},
    )
    assert gate_set | GateSet({ToffoliGate()}) == GateSet(
        {CNOTGate(), U3Gate(), ToffoliGate()},
    )
    assert gate_set | GateSet({}) == gate_set


def test_gate_set_or_other_set() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set | {CNOTGate(), U3Gate()} == gate_set
    assert gate_set | {CNOTGate()} == gate_set
    assert gate_set | {U3Gate()} == gate_set
    assert gate_set | {CNOTGate(), U3Gate(), ToffoliGate()} == GateSet({
        CNOTGate(), U3Gate(), ToffoliGate(),
    })
    assert gate_set | {ToffoliGate()} == GateSet(
        {CNOTGate(), U3Gate(), ToffoliGate()},
    )
    assert gate_set | set() == gate_set


def test_gate_set_xor() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set ^ gate_set == GateSet({})
    assert gate_set ^ GateSet({CNOTGate(), U3Gate()}) == GateSet({})
    assert gate_set ^ GateSet({CNOTGate()}) == GateSet({U3Gate()})
    assert gate_set ^ GateSet({U3Gate()}) == GateSet({CNOTGate()})
    assert gate_set ^ GateSet(
        {CNOTGate(), U3Gate(), ToffoliGate()},
    ) == GateSet({ToffoliGate()})
    assert gate_set ^ GateSet({ToffoliGate()}) == GateSet(
        {CNOTGate(), U3Gate(), ToffoliGate()},
    )
    assert gate_set ^ GateSet({}) == gate_set


def test_gate_set_xor_other_set() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set ^ {CNOTGate(), U3Gate()} == GateSet({})
    assert gate_set ^ {CNOTGate()} == GateSet({U3Gate()})
    assert gate_set ^ {U3Gate()} == GateSet({CNOTGate()})
    assert gate_set ^ {
        CNOTGate(), U3Gate(), ToffoliGate(),
    } == GateSet({ToffoliGate()})
    assert gate_set ^ {ToffoliGate()} == GateSet(
        {CNOTGate(), U3Gate(), ToffoliGate()},
    )
    assert gate_set ^ set() == gate_set


def test_gate_set_sub() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set - gate_set == GateSet({})
    assert gate_set - GateSet({CNOTGate(), U3Gate()}) == GateSet({})
    assert gate_set - GateSet({CNOTGate()}) == GateSet({U3Gate()})
    assert gate_set - GateSet({U3Gate()}) == GateSet({CNOTGate()})
    assert gate_set - \
        GateSet({CNOTGate(), U3Gate(), ToffoliGate()}) == GateSet({})
    assert gate_set - GateSet({ToffoliGate()}) == gate_set
    assert gate_set - GateSet({}) == gate_set


def test_gate_set_sub_other_set() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert gate_set - {CNOTGate(), U3Gate()} == GateSet({})
    assert gate_set - {CNOTGate()} == GateSet({U3Gate()})
    assert gate_set - {U3Gate()} == GateSet({CNOTGate()})
    assert gate_set - {CNOTGate(), U3Gate(), ToffoliGate()} == GateSet({})
    assert gate_set - {ToffoliGate()} == gate_set
    assert gate_set - set() == gate_set


def test_gate_set_set_update() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    x: set[Gate] = set()
    x.update(gate_set)
    assert x == {CNOTGate(), U3Gate()}
    assert x == gate_set


def test_gate_set_str() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert (
        str(gate_set) == 'GateSet({CNOTGate, U3Gate})'
        or str(gate_set) == 'GateSet({U3Gate, CNOTGate})'
    )


def test_gate_set_repr() -> None:
    gate_set = GateSet({CNOTGate(), U3Gate()})
    assert (
        repr(gate_set) == 'GateSet({CNOTGate, U3Gate})'
        or repr(gate_set) == 'GateSet({U3Gate, CNOTGate})'
    )
