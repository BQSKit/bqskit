"""This module tests the PowerGate class."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates import DaggerGate
from bqskit.ir.gates import PowerGate
from bqskit.ir.gates import RXGate
from bqskit.ir.gates import RYGate
from bqskit.ir.gates import RZGate


test_power = lambda gate, power, params: np.linalg.matrix_power(
    gate.get_unitary([params]), power,
)


def square_grad(gate, params):
    g, gd = gate.get_unitary_and_grad([params])
    return g @ gd + gd @ g


def third_power_grad(gate, params):
    g, gd = gate.get_unitary_and_grad([params])
    return g @ square_grad(gate, params) + gd @ test_power(gate, 2, params)


def quartic_power_grad(gate, params):
    g, gd = gate.get_unitary_and_grad([params])
    return g @ third_power_grad(gate, params) + gd @ test_power(gate, 3, params)


def power_gate_grads(gate, power, params):
    if power == 2:
        return square_grad(gate, params)
    elif power == -2:
        return square_grad(DaggerGate(gate), params)
    elif power == 3:
        return third_power_grad(gate, params)
    elif power == -3:
        return third_power_grad(DaggerGate(gate), params)
    elif power == 4:
        return quartic_power_grad(gate, power)
    elif power == -4:
        return quartic_power_grad(DaggerGate(gate), power)


def test(test_gate, indices, params, error) -> None:

    # test index 1
    for param in params:
        pgt, pgdt = test_gate.get_unitary_and_grad([param])
        pgate = PowerGate(test_gate, 1)
        pg, pgd = pgate.get_unitary_and_grad([param])
        assert np.sum(abs(pg - pgt)) < error
        assert np.sum(abs(pgd - pgdt)) < error

    # test index -1
    for param in params:
        pgt, pgdt = DaggerGate(test_gate).get_unitary_and_grad([param])
        pgate = PowerGate(test_gate, -1)
        pg, pgd = pgate.get_unitary_and_grad([param])
        assert np.sum(abs(pg - pgt)) < error
        assert np.sum(abs(pgd - pgdt)) < error

    # test other indices
    for index in indices:
        for param in params:
            gate = test_power(test_gate, index, param)
            grad = power_gate_grads(test_gate, index, param)

            pgate = PowerGate(test_gate, index)
            pg, pgd = pgate.get_unitary_and_grad([param])
            assert np.sum(abs(pg - gate)) < error
            assert np.sum(abs(pgd - grad)) < error


error = 1e-14
params = [-0.7, -0.3, 0.2, 1.4]
indices = [-4, -3, -2, 2, 3, 4]


def test_x() -> None:
    global error, indices, parames
    test(RXGate(), indices, params, error)


def test_y() -> None:
    global error, indices, parames
    test(RYGate(), indices, params, error)


def test_z() -> None:
    global error, indices, parames
    test(RZGate(), indices, params, error)
