from __future__ import annotations

import numpy as np

from bqskit.ir import Circuit
from bqskit.ir.gates.constant import XGate
from bqskit.ir.gates.parameterized import RXGate
from bqskit.ir.opt.cost import HilbertSchmidtCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


def test_minimize_bfgs() -> None:
    circ = Circuit(1)
    circ.append_gate(RXGate(), location=[0], params=[0.0])
    xgate = XGate()
    xutry = xgate.get_unitary()
    cost = HilbertSchmidtCostGenerator().gen_cost(
        circ, UnitaryMatrix(-1j * xutry.get_numpy()),
    )
    minimizer = LBFGSMinimizer()
    x = minimizer.minimize(cost, np.array([np.pi / 2]))
    assert cost.get_cost(x) < 1e-6, x
