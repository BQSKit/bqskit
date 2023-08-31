from __future__ import annotations

from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates.constant.cpi import CPIGate
from bqskit.ir.gates.parameterized.u8 import U8Gate
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes.search.generators.seed import SeedLayerGenerator
from bqskit.passes.search.generators.simple import SimpleLayerGenerator
from bqskit.qis import UnitaryMatrix


class TestQSearch:

    def test_small_qubit(self, compiler: Compiler) -> None:
        utry = UnitaryMatrix.random(2)
        circuit = Circuit.from_unitary(utry)
        qsearch = QSearchSynthesisPass()
        circuit = compiler.compile(circuit, [qsearch])
        dist = circuit.get_unitary().get_distance_from(utry, 1)
        assert dist <= 1e-5

    def test_small_qutrit(self) -> None:
        utry = UnitaryMatrix.random(2, [3, 3])
        circuit = Circuit.from_unitary(utry)
        layer_gen = SimpleLayerGenerator(CPIGate(), U8Gate())
        leap = QSearchSynthesisPass(layer_generator=layer_gen)
        circuit.perform(leap)
        dist = circuit.get_unitary().get_distance_from(utry, 1)
        assert dist <= leap.success_threshold

    def test_seed(self) -> None:
        seed = Circuit(2)
        seed.append_gate(CNOTGate(), (0, 1))
        layer_gen = SeedLayerGenerator(seed)
        qsearch = QSearchSynthesisPass(layer_generator=layer_gen)
        utry = UnitaryMatrix.random(2)
        circuit = Circuit.from_unitary(utry)
        circuit.perform(qsearch)
        dist = circuit.get_unitary().get_distance_from(utry, 1)
        assert dist <= 1e-5

    def test_seed_in_data(self) -> None:
        seed = Circuit(2)
        seed.append_gate(CNOTGate(), (0, 1))
        qsearch = QSearchSynthesisPass()
        utry = UnitaryMatrix.random(2)
        circuit = Circuit.from_unitary(utry)
        data = {'seed_circuits': [seed]}
        circuit.perform(qsearch, data)
        dist = circuit.get_unitary().get_distance_from(utry, 1)
        assert dist <= 1e-5
