# type: ignore
"""This module tests the PowerGate class."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from hypothesis import given
from hypothesis.strategies import integers

from bqskit.ir.gate import Gate
from bqskit.ir.gates import PowerGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.test.strategies import gates_and_params


def _recursively_calc_power_grad(
    g: UnitaryMatrix,
    dg: npt.NDArray[np.complex128],
    power: int,
) -> npt.NDArray[np.complex128]:
    """D(g^n+1) = d(g@g^n) = g @ d(g^n) + dg @ g^n."""
    if len(dg) == 0 or power == 0:
        return np.zeros_like(dg)
    if power < 0:
        return _recursively_calc_power_grad(
            g.dagger,
            dg.conj().transpose([0, 2, 1]),
            -power,
        )
    if power == 1:
        return dg
    dgn = _recursively_calc_power_grad(g, dg, power - 1)
    return g @ dgn + dg @ g.ipower(power - 1)

from bqskit.ir.gates import CRYGate
@given(gates_and_params(), integers(min_value=-10, max_value=10))
def test_power_gate(g_and_p: tuple[Gate, RealVector], power: int) -> None:
    gate, params = g_and_p
    pgate = PowerGate(gate, power)
    actual_unitary = pgate.get_unitary(params)
    expected_unitary = gate.get_unitary(params).ipower(power)
    assert actual_unitary.isclose(expected_unitary)

    if not isinstance(gate, DifferentiableUnitary):
        return

    if gate.num_params == 0:
        return

    actual_grad = pgate.get_grad(params)
    expected_grad = _recursively_calc_power_grad(
        gate.get_unitary(params),
        gate.get_grad(params),
        power,
    )
    assert np.allclose(actual_grad, expected_grad)
