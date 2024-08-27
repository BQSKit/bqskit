from __future__ import annotations

from random import randint

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import TGate
from bqskit.passes.search.generators import DiscreteLayerGenerator


class TestDiscreteLayerGenerator:

    def test_gate_set(self) -> None:
        gates = [HGate(), CNOTGate(), TGate()]
        generator = DiscreteLayerGenerator()
        assert all(g in generator.gateset for g in gates)

    def test_double_headed(self) -> None:
        single_gen = DiscreteLayerGenerator(double_headed=False)
        double_gen = DiscreteLayerGenerator(double_headed=True)
        base = Circuit(4)
        single_sucs = single_gen.gen_successors(base, PassData(base))
        double_sucs = double_gen.gen_successors(base, PassData(base))
        assert len(single_sucs) == len(double_sucs)

        base = Circuit(2)
        base.append_gate(CNOTGate(), (0, 1))
        single_sucs = single_gen.gen_successors(base, PassData(base))
        double_sucs = double_gen.gen_successors(base, PassData(base))
        assert len(single_sucs) < len(double_sucs)
        assert all(c in double_sucs for c in single_sucs)

    def test_cancels_something(self) -> None:
        gen = DiscreteLayerGenerator()
        base = Circuit(2)
        base.append_gate(HGate(), (0,))
        base.append_gate(TGate(), (0,))
        base.append_gate(HGate(), (0,))
        assert gen.cancels_something(base, HGate(), (0,))
        assert not gen.cancels_something(base, HGate(), (1,))
        assert not gen.cancels_something(base, TGate(), (0,))

    def test_count_repeats(self) -> None:
        num_repeats = randint(1, 50)
        c = Circuit(1)
        for _ in range(num_repeats):
            c.append_gate(HGate(), (0,))
        gen = DiscreteLayerGenerator()
        assert gen.count_repeats(c, HGate(), 0) == num_repeats
