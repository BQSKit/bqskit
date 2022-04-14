"""This module tests the StateVector class."""
from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis.strategies import floats

from bqskit.qis.state.state import StateLike
from bqskit.qis.state.state import StateVector
from bqskit.utils.test.strategies import num_qudits_and_radixes
from bqskit.utils.test.strategies import state_likes
from bqskit.utils.test.strategies import state_vectors


class TestInit:

    @given(state_vectors())
    def test_init_copy(self, vec: StateVector) -> None:
        new_vec = StateVector(vec)
        assert new_vec.radixes == vec.radixes
        assert new_vec == vec

    @given(state_vectors())
    def test_init(self, vec: StateVector) -> None:
        new_vec = StateVector(vec.numpy, vec.radixes)
        assert new_vec.radixes == vec.radixes
        assert new_vec == vec


@given(state_vectors())
def test_properties(vec: StateVector) -> None:
    assert vec.shape == (vec.dim,)
    assert vec.dtype == np.complex128
    assert vec == vec.numpy
    assert len(vec) == vec.dim
    assert vec.num_qudits == len(vec.radixes)
    assert int(np.prod(vec.radixes)) == vec.dim


@given(state_likes(3))
def test_is_unitary(v: StateLike) -> None:
    assert StateVector.is_pure_state(v)
    assert not StateVector.is_pure_state(np.array(v) + 5)


@given(num_qudits_and_radixes(3))
def test_random(pair: tuple[int, tuple[int, ...]]) -> None:
    num_qudits, radixes = pair
    v = StateVector.random(num_qudits, radixes)
    assert isinstance(v, StateVector)


class TestClosedOperations:

    @given(state_vectors())
    def test_conjugate(self, v: StateVector) -> None:
        out = np.conjugate(v)
        assert out is not v
        assert isinstance(out, StateVector)

    @given(state_vectors(), floats(allow_nan=False, allow_infinity=False))
    def test_scalar_multiplication(self, v: StateVector, a: float) -> None:
        out = np.exp(0 + 1j * a) * v
        assert out is not v
        assert isinstance(out, StateVector)

        if np.abs(a) != 1:
            out2 = a * v
            assert out2 is not v
            assert not isinstance(out2, StateVector)
