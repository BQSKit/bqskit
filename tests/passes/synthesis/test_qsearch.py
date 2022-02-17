from __future__ import annotations

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CPIGate
from bqskit.ir.gates import U8Gate
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import SimpleLayerGenerator
from bqskit.qis import UnitaryMatrix


class TestQSearch:

    def test_small_qubit(self) -> None:
        utry = UnitaryMatrix.random(2)
        circuit = Circuit.from_unitary(utry)
        qsearch = QSearchSynthesisPass()
        qsearch.run(circuit)
        dist = circuit.get_unitary().get_distance_from(utry)
        assert dist <= 1e-5

    def test_small_qutrit(self) -> None:
        utry = UnitaryMatrix.random(2, [3, 3])
        circuit = Circuit.from_unitary(utry)
        layer_gen = SimpleLayerGenerator(CPIGate(), U8Gate())
        qsearch = QSearchSynthesisPass(layer_generator=layer_gen)
        qsearch.run(circuit)
        dist = circuit.get_unitary().get_distance_from(utry)
        assert dist <= qsearch.success_threshold
