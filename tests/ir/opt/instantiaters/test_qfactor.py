"""This test module verifies the QFactor instantiater."""
from __future__ import annotations

import numpy as np
from scipy.stats import unitary_group

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized import RXGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.opt.instantiaters.qfactor import QFactor


class TestQFactorEndToEnd:

    def test_no_change(self) -> None:
        u1 = unitary_group.rvs(8)
        g1 = VariableUnitaryGate(3)
        circuit = Circuit(3)
        circuit.append_gate(g1, [0, 1, 2])
        utry_before = circuit.get_unitary()
        # The following call should not make any changes in circuit
        QFactor().instantiate(circuit, u1, circuit.get_params())
        utry_after = circuit.get_unitary()

        assert np.allclose(
            utry_before.get_numpy(),
            utry_after.get_numpy(),
        )

    def test_1_gate(self) -> None:
        u1 = unitary_group.rvs(8)
        g1 = VariableUnitaryGate(3)
        circuit = Circuit(3)
        circuit.append_gate(g1, [0, 1, 2])
        params = QFactor().instantiate(circuit, u1, circuit.get_params())
        circuit.set_params(params)

        g1_params = list(np.reshape(u1, (64,)))
        g1_params = list(np.real(g1_params)) + list(np.imag(g1_params))

        assert np.allclose(
            circuit.get_unitary().get_numpy(),
            g1.get_unitary(g1_params).get_numpy(),
        )

    def test_2_gate(self) -> None:
        g1 = VariableUnitaryGate(2)
        g2 = VariableUnitaryGate(3)
        g3 = RXGate()
        circuit = Circuit(4)
        circuit.append_gate(g1, [0, 1])
        circuit.append_gate(g2, [1, 2, 3])
        circuit.append_gate(g3, [1])
        utry = circuit.get_unitary(np.random.random(circuit.get_num_params()))
        params = QFactor().instantiate(circuit, utry, circuit.get_params())

        circuit.set_params(params)

        assert np.allclose(
            circuit.get_unitary().get_numpy(),
            utry.get_numpy(),
        )
