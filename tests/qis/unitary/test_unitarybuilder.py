"""This module tests the UnitaryBuilder class in bqskit.qis.unitary."""
from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np
import pytest

from bqskit.qis.unitary.unitarybuilder import UnitaryBuilder
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class TestConstructor:

    def test_invalid_type_size(self, not_an_int: Any) -> None:
        with pytest.raises(TypeError):
            UnitaryBuilder(not_an_int)

    def test_invalid_type_radixes(self, not_a_seq_int: Any) -> None:
        if not_a_seq_int == '':
            return
        with pytest.raises(TypeError):
            UnitaryBuilder(5, not_a_seq_int)

    @pytest.mark.parametrize('size', [0, -1, -10000])
    def test_invalid_value_size(self, size: int) -> None:
        with pytest.raises(ValueError):
            UnitaryBuilder(size)

    @pytest.mark.parametrize('radixes', [(0,), [0, 1, 2], [-1, -1]])
    def test_invalid_value_radixes(self, radixes: Sequence[int]) -> None:
        with pytest.raises(TypeError):
            UnitaryBuilder(2, radixes)

    @pytest.mark.parametrize(
        ['size', 'radixes'],
        [
            (2, []),
            (4, []),
            (3, [2, 2, 2]),
            (3, [2, 3, 3]),
            (3, [3, 3, 3]),
            (3, [2, 3, 4]),
        ],
    )
    def test_valid(self, size: int, radixes: Sequence[int]) -> None:
        ub = UnitaryBuilder(size, radixes)
        assert ub.num_qudits == size
        assert isinstance(ub.radixes, tuple)
        assert len(ub.radixes) == size
        if len(radixes) > 0:
            for radix1, radix2 in zip(radixes, ub.radixes):
                assert radix1 == radix2
        assert ub.dim == int(np.prod(ub.radixes))
        assert np.allclose(
            ub.get_unitary(),
            np.identity(ub.dim),
        )


class TestApplyLeft:

    def test_valid_1(self) -> None:
        u1 = UnitaryMatrix.random(3)
        ub = UnitaryBuilder(3)

        ub.apply_left(u1, [0, 1, 2])
        assert ub.get_unitary() == u1

    def test_valid_2(self) -> None:
        u1 = UnitaryMatrix.random(3)
        u2 = UnitaryMatrix.random(2)
        ub = UnitaryBuilder(3)

        ub.apply_left(u1, [0, 1, 2])
        assert ub.get_unitary() == u1
        ub.apply_left(u2, [0, 1])
        prod = u1 @ np.kron(u2, np.identity(2))
        assert ub.get_unitary() == prod

    def test_valid_3(self) -> None:
        u1 = UnitaryMatrix.random(3)
        u2 = UnitaryMatrix.random(2)
        ub = UnitaryBuilder(3)

        ub.apply_left(u1, [0, 1, 2])
        assert ub.get_unitary() == u1
        ub.apply_left(u2, [1, 2])
        prod = u1 @ np.kron(np.identity(2), u2)
        assert ub.get_unitary() == prod


class TestApplyRight:

    def test_valid_1(self) -> None:
        u1 = UnitaryMatrix.random(3)
        ub = UnitaryBuilder(3)

        ub.apply_right(u1, [0, 1, 2])
        assert ub.get_unitary() == u1

    def test_valid_2(self) -> None:
        u1 = UnitaryMatrix.random(3)
        u2 = UnitaryMatrix.random(2)
        ub = UnitaryBuilder(3)

        ub.apply_right(u1, [0, 1, 2])
        assert ub.get_unitary() == u1
        ub.apply_right(u2, [0, 1])
        prod = np.kron(u2, np.identity(2)) @ u1
        assert ub.get_unitary() == prod

    def test_valid_3(self) -> None:
        u1 = UnitaryMatrix.random(3)
        u2 = UnitaryMatrix.random(2)
        ub = UnitaryBuilder(3)

        ub.apply_right(u1, [0, 1, 2])
        assert ub.get_unitary() == u1
        ub.apply_right(u2, [1, 2])
        prod = np.kron(np.identity(2), u2) @ u1
        assert ub.get_unitary() == prod


class TestCalcEnvMatrix:

    def test_valid_1(self) -> None:
        u1 = UnitaryMatrix.random(3)
        ub = UnitaryBuilder(3)

        ub.apply_right(u1, [0, 1, 2])
        assert u1 == ub.calc_env_matrix([0, 1, 2])
