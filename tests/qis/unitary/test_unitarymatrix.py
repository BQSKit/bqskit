"""This module tests the UnitaryMatrix class in bqskit.qis.unitary."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import scipy
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import complex_numbers
from hypothesis.strategies import floats
from hypothesis.strategies import integers

from bqskit.qis.state.state import StateLike
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.test.strategies import num_qudits
from bqskit.utils.test.strategies import num_qudits_and_radixes
from bqskit.utils.test.strategies import radixes
from bqskit.utils.test.strategies import state_likes
from bqskit.utils.test.strategies import unitaries
from bqskit.utils.test.strategies import unitary_likes


class TestInit:
    @given(unitaries())
    def test_init_copy(self, utry: UnitaryMatrix) -> None:
        new_utry = UnitaryMatrix(utry)
        assert new_utry.radixes == utry.radixes
        assert new_utry == utry

    @given(unitaries())
    def test_init(self, utry: UnitaryMatrix) -> None:
        new_utry = UnitaryMatrix(utry.numpy, utry.radixes)
        assert new_utry.radixes == utry.radixes
        assert new_utry == utry


@given(unitaries())
def test_properties(utry: UnitaryMatrix) -> None:
    assert utry.shape == (utry.dim, utry.dim)
    assert utry.dtype == np.complex128
    assert utry == utry.numpy
    assert utry.T == utry.numpy.T
    assert utry.dagger == utry.numpy.T.conj()
    assert len(utry) == utry.dim


@given(unitaries())
def test_conj(utry: UnitaryMatrix) -> None:
    assert utry.conj() == utry.numpy.conj()


class TestOtimes:
    @given(unitaries(2), unitaries(2))
    def test_two(self, u1: UnitaryMatrix, u2: UnitaryMatrix) -> None:
        kron = u1.otimes(u2)
        assert isinstance(kron, UnitaryMatrix)
        assert len(kron.radixes) == len(u1.radixes) + len(u2.radixes)
        assert kron.radixes[:len(u1.radixes)] == u1.radixes
        assert kron.radixes[len(u1.radixes):] == u2.radixes
        assert np.allclose(kron.numpy, np.kron(u1, u2))

    @given(unitaries(2), unitaries(2), unitaries(1))
    def test_three(
        self,
        u1: UnitaryMatrix,
        u2: UnitaryMatrix,
        u3: UnitaryMatrix,
    ) -> None:
        kron = u1.otimes(u2, u3)
        assert isinstance(kron, UnitaryMatrix)
        total_qudits = len(u1.radixes) + len(u2.radixes) + len(u3.radixes)
        assert len(kron.radixes) == total_qudits
        assert kron.radixes[:len(u1.radixes)] == u1.radixes
        sep = len(u1.radixes) + len(u2.radixes)
        assert kron.radixes[len(u1.radixes):sep] == u2.radixes
        assert kron.radixes[sep:] == u3.radixes
        assert np.allclose(kron.numpy, np.kron(np.kron(u1, u2), u3))


@given(unitaries())
def test_get_unitary(utry: UnitaryMatrix) -> None:
    assert utry.get_unitary() is utry


class TestGetDistanceFrom:
    @given(unitaries(1, (2,)), unitaries(1, (2,)))
    def test_get_distance_from(
        self, u1: UnitaryMatrix,
        u2: UnitaryMatrix,
    ) -> None:
        assert u1.get_distance_from(u2) == u2.get_distance_from(u1)
        assert 0 <= u1.get_distance_from(u2) <= 1

    @given(unitaries())
    def test_same(self, utry: UnitaryMatrix) -> None:
        assert utry.get_distance_from(utry) <= 1e-7

    @given(unitaries(), floats(allow_nan=False, allow_infinity=False))
    def test_global_phase(self, utry: UnitaryMatrix, angle: float) -> None:
        u1 = np.exp(0 + 1j * angle) * UnitaryMatrix(utry)
        assert utry.get_distance_from(u1) <= 1e-7

    def test_numpy(self) -> None:
        utry = UnitaryMatrix.random(2)
        np_utry = scipy.stats.unitary_group.rvs(4)
        assert isinstance(utry.get_distance_from(np_utry), float)


class TestGetStateVector:
    @given(unitaries(1, (2,)), state_likes(1, (2,)))
    def test_simple(self, u: UnitaryMatrix, v: StateLike) -> None:
        o = u.get_statevector(v)
        v = v.numpy if isinstance(v, StateVector) else np.array(v)
        assert np.allclose(o, (u.numpy @ v[:, None]).reshape((-1,)))
        assert tuple(o.shape) == tuple(v.shape)

    @given(num_qudits_and_radixes(3, (2, 3)))
    def test_radixes(self, pair: tuple[int, tuple[int, ...]]) -> None:
        num_qudits, radixes = pair
        u = UnitaryMatrix.random(num_qudits, radixes)
        v = StateVector.random(num_qudits, radixes)
        o = u.get_statevector(v)
        assert np.allclose(o, (u.numpy @ v.numpy[:, None]).reshape((-1,)))
        assert tuple(o.shape) == tuple(v.shape)


class TestIdentity:
    @given(radixes())
    def test_identity(self, radixes: tuple[int, ...]) -> None:
        utry = UnitaryMatrix.identity(int(np.prod(radixes)), radixes)
        assert utry == np.identity(int(np.prod(radixes)))
        assert utry.radixes == radixes

    @given(num_qudits())
    def test_identity_no_radixes(self, num_qudits: int) -> None:
        utry = UnitaryMatrix.identity(2 ** num_qudits)
        assert utry == np.identity(2 ** num_qudits)
        assert utry.radixes == tuple([2] * num_qudits)

    @given(integers(max_value=0))
    def test_invalid_value(self, dim: int) -> None:
        with pytest.raises(ValueError):
            UnitaryMatrix.identity(dim)


@given(
    arrays(
        np.complex128,
        (4, 4),
        elements=complex_numbers(allow_infinity=False, allow_nan=False),
    ),
)
def test_closest_to(M: npt.NDArray[np.complex128]) -> None:
    u = UnitaryMatrix.closest_to(M, (2, 2))
    assert isinstance(u, UnitaryMatrix)


@given(num_qudits_and_radixes(3))
def test_random(pair: tuple[int, tuple[int, ...]]) -> None:
    num_qudits, radixes = pair
    u = UnitaryMatrix.random(num_qudits, radixes)
    assert isinstance(u, UnitaryMatrix)


@given(unitary_likes(3))
def test_is_unitary(u: UnitaryLike) -> None:
    assert UnitaryMatrix.is_unitary(u)
    assert not UnitaryMatrix.is_unitary(np.array(u) + 5)


class TestClosedOperations:
    @given(num_qudits_and_radixes(3, (2, 3)))
    def test_matmul(self, pair: tuple[int, tuple[int, ...]]) -> None:
        num_qudits, radixes = pair
        u1 = UnitaryMatrix.random(num_qudits, radixes)
        u2 = UnitaryMatrix.random(num_qudits, radixes)
        out = u1 @ u2
        assert out is not u1
        assert out is not u2
        assert isinstance(out, UnitaryMatrix)

        out2 = u1 @ u2.numpy
        assert not isinstance(out2, UnitaryMatrix)

        out3 = u1.numpy @ u2
        assert not isinstance(out3, UnitaryMatrix)

    @given(unitaries())
    def test_conjugate(self, u: UnitaryMatrix) -> None:
        out = np.conjugate(u)
        assert out is not u
        assert isinstance(out, UnitaryMatrix)

    @given(unitaries())
    def test_negative(self, u: UnitaryMatrix) -> None:
        out = np.negative(u)
        assert out is not u
        assert isinstance(out, UnitaryMatrix)

    @given(unitaries())
    def test_positive(self, u: UnitaryMatrix) -> None:
        out = np.positive(u)
        assert out is not u
        assert isinstance(out, UnitaryMatrix)

    @given(unitaries(), floats(allow_nan=False, allow_infinity=False))
    def test_scalar_multiplication(self, u: UnitaryMatrix, a: float) -> None:
        out = np.exp(0 + 1j * a) * u
        assert out is not u
        assert isinstance(out, UnitaryMatrix)

        if np.abs(a) != 1:
            out2 = a * u
            assert out2 is not u
            assert not isinstance(out2, UnitaryMatrix)
