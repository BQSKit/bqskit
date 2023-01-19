"""This module tests the Operation class."""
from __future__ import annotations

import numpy as np
from hypothesis import assume
from hypothesis import given

from bqskit.ir.gates import CXGate
from bqskit.ir.gates import CYGate
from bqskit.ir.gates import U3Gate
from bqskit.ir.operation import Operation
from bqskit.utils.test.strategies import operations


@given(operations())
def test_init(op: Operation) -> None:
    new_op = Operation(op.gate, op.location, op.params)
    assert new_op.gate == op.gate
    assert new_op.location == op.location
    assert new_op.params == op.params
    assert new_op.radixes == op.radixes
    assert new_op.get_unitary() == op.get_unitary()


class TestGetQasm:
    def test_cx(self) -> None:
        op = Operation(CXGate(), (0, 1))
        assert op.get_qasm() == 'cx q[0], q[1];\n'

    def test_cy(self) -> None:
        op = Operation(CYGate(), (3, 0))
        assert op.get_qasm() == 'cy q[3], q[0];\n'

    def test_u3(self) -> None:
        op = Operation(U3Gate(), 0, [0, 1, 2])
        assert op.get_qasm() == 'u3(0, 1, 2) q[0];\n'


@given(operations())
def test_get_unitary(op: Operation) -> None:
    assert op.get_unitary() == op.gate.get_unitary(op.params)

    new_params = [1] * op.num_params
    assert op.get_unitary(new_params) == op.get_unitary(new_params)


@given(operations())
def test_get_grad(op: Operation) -> None:
    assume(op.is_differentiable())
    assert np.allclose(op.get_grad(), op.gate.get_grad(op.params))  # type: ignore  # noqa

    new_params = [1] * op.num_params
    assert np.allclose(op.get_grad(new_params), op.gate.get_grad(new_params))  # type: ignore  # noqa


@given(operations())
def test_get_unitary_and_grad(op: Operation) -> None:
    assume(op.is_differentiable())
    utry, grads = op.get_unitary_and_grad()
    exp_utry, exp_grads = op.gate.get_unitary_and_grad(op.params)  # type: ignore  # noqa
    assert utry == exp_utry
    assert np.allclose(grads, exp_grads)

    new_params = [1] * op.num_params
    utry, grads = op.get_unitary_and_grad(new_params)
    exp_utry, exp_grads = op.gate.get_unitary_and_grad(new_params)  # type: ignore  # noqa
    assert utry == exp_utry
    assert np.allclose(grads, exp_grads)
