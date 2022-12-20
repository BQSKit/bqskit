"""This module tests the CircuitPoint class."""
from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.strategies import tuples

from bqskit.ir.point import CircuitPoint
from bqskit.ir.point import CircuitPointLike
from bqskit.utils.test.strategies import circuit_point_likes
from bqskit.utils.test.strategies import circuit_points
from bqskit.utils.test.types import invalid_type_test
from bqskit.utils.test.types import type_annotation_to_invalid_strategy
from bqskit.utils.test.types import valid_type_test


class TestNew:
    @invalid_type_test(CircuitPoint)
    def test_invalid_type(self) -> None:
        pass

    @valid_type_test(CircuitPoint)
    def test_valid_type(self) -> None:
        pass

    @given(tuples(integers(), integers()))
    def test_from_tuple(self, pair: tuple[int, int]) -> None:
        point = CircuitPoint(pair)
        assert point == pair
        assert point[0] == pair[0]
        assert point[1] == pair[1]
        assert len(point) == len(pair)
        assert isinstance(point, tuple)
        assert point.cycle == pair[0]
        assert point.qudit == pair[1]

    @given(integers(), integers())
    def test_from_ints(self, cycle: int, qudit: int) -> None:
        point = CircuitPoint(cycle, qudit)
        assert point == (cycle, qudit)
        assert point[0] == cycle
        assert point[1] == qudit
        assert len(point) == 2
        assert isinstance(point, tuple)
        assert point.cycle == cycle
        assert point.qudit == qudit

    @given(integers())
    def test_invalid(self, cycle: int) -> None:
        with pytest.raises(ValueError):
            CircuitPoint(cycle)


class TestIsPoint:
    @given(circuit_points())
    def test_from_point(self, point: CircuitPoint) -> None:
        assert CircuitPoint.is_point(point)

    @given(circuit_point_likes())
    def test_from_like(self, point: CircuitPointLike) -> None:
        assert CircuitPoint.is_point(point)

    @given(type_annotation_to_invalid_strategy('Tuple[int, int]'))
    def test_false(self, not_a_point: Any) -> None:
        assert not CircuitPoint.is_point(not_a_point)


class TestConversionToTuple:
    @given(circuit_points())
    def test_from_point(self, point: CircuitPoint) -> None:
        assert isinstance(point, tuple)
        assert isinstance(point, CircuitPoint)
        assert not isinstance(tuple(point), CircuitPoint)
