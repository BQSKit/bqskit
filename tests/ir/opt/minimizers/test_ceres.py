from __future__ import annotations

import numpy as np

from bqskit.ir import Circuit
from bqskit.ir.gates.constant import XGate
from bqskit.ir.gates.parameterized import RXGate
from bqskit.ir.opt.cost import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.minimizers.ceres import CeresMinimizer
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


def test_minimize_ceres() -> None:
    circ = Circuit(1)
    circ.append_gate(RXGate(), location=[0], params=[0.0])
    xgate = XGate()
    xutry = xgate.get_unitary()
    cost = HilbertSchmidtResidualsGenerator().gen_cost(
        circ, UnitaryMatrix(-1j * xutry),
    )
    minimizer = CeresMinimizer()
    x = minimizer.minimize(cost, np.array([np.pi / 2]))
    assert cost.get_cost(x) < 1e-6, x


def test_minimize_bfgs_multiqubit(r3_qubit_circuit: Circuit) -> None:
    num_params = r3_qubit_circuit.get_num_params()
    x0 = np.random.random((num_params,))
    cost = HilbertSchmidtResidualsGenerator().gen_cost(
        r3_qubit_circuit, r3_qubit_circuit.get_unitary(x0),
    )
    minimizer = CeresMinimizer()
    x = minimizer.minimize(cost, np.random.random((num_params,)))
    assert cost.get_cost(x) < 1e-6, x
