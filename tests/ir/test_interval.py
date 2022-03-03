"""This module tests the CycleInterval class."""
from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.strategies import tuples

from bqskit.ir.interval import CycleInterval
from bqskit.utils.test.strategies import cycle_intervals
from bqskit.utils.test.strategies import everything_except
from bqskit.utils.test.types import invalid_type_test
from bqskit.utils.test.types import type_annotation_to_invalid_strategy
from bqskit.utils.test.types import valid_type_test
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_iterable
from bqskit.utils.typing import is_sequence_of_int


@given(cycle_intervals())
def test_strategy(interval: CycleInterval) -> None:
    assert isinstance(interval, CycleInterval)
    assert CycleInterval.is_interval(interval)


class TestCycleIntervalNew:

    @valid_type_test(CycleInterval)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(CycleInterval)
    def test_invalid_type(self) -> None:
        pass

    @given(tuples(integers(), integers()).filter(lambda x: x[1] < x[0]))
    def test_invalid_value(self, interval: tuple[int, int]) -> None:
        with pytest.raises(ValueError):
            CycleInterval(interval)
        with pytest.raises(ValueError):
            CycleInterval(*interval)

    @given(tuples(integers(), integers()).filter(lambda x: 0 <= x[0] <= x[1]))
    def test_valid_1(self, interval: tuple[int, int]) -> None:
        interval_obj = CycleInterval(interval)
        assert interval_obj.lower == interval[0]
        assert interval_obj.upper == interval[1]
        assert interval_obj[0] == interval[0]
        assert interval_obj[1] == interval[1]
        assert isinstance(interval_obj, tuple)
        assert isinstance(interval_obj[0], int)
        assert isinstance(interval_obj[1], int)

    @given(tuples(integers(), integers()).filter(lambda x: 0 <= x[0] <= x[1]))
    def test_valid_2(self, interval: tuple[int, int]) -> None:
        interval_obj = CycleInterval(*interval)
        assert interval_obj.lower == interval[0]
        assert interval_obj.upper == interval[1]
        assert interval_obj[0] == interval[0]
        assert interval_obj[1] == interval[1]
        assert isinstance(interval_obj, tuple)
        assert isinstance(interval_obj[0], int)
        assert isinstance(interval_obj[1], int)


@given(cycle_intervals())
def test_lower(interval: CycleInterval) -> None:
    assert interval.lower == interval[0]
    assert isinstance(interval.lower, int)


@given(cycle_intervals())
def test_upper(interval: CycleInterval) -> None:
    assert interval.upper == interval[1]
    assert isinstance(interval.upper, int)


@given(cycle_intervals())
def test_iter(interval: CycleInterval) -> None:
    assert min(list(interval)) == interval.lower
    assert max(list(interval)) == interval.upper
    assert is_iterable(interval)
    assert all(is_integer(idx) for idx in interval)


@given(cycle_intervals())
def test_len(interval: CycleInterval) -> None:
    assert len(interval) == len(list(interval))


@given(cycle_intervals())
def test_indices(interval: CycleInterval) -> None:
    indices = interval.indices
    assert len(indices) == interval.upper - interval.lower + 1
    assert min(indices) == interval.lower
    assert max(indices) == interval.upper
    assert is_sequence_of_int(indices)


class TestCycleIntervalOverlaps:

    @valid_type_test(CycleInterval(0, 0).overlaps)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(CycleInterval(0, 0).overlaps)
    def test_invalid_type(self) -> None:
        pass

    @given(cycle_intervals(), cycle_intervals())
    def test_overlaps(
        self,
        interval1: CycleInterval,
        interval2: CycleInterval,
    ) -> None:
        if interval1.overlaps(interval2):
            assert any(x in interval2 for x in interval1)
        else:
            assert all(x not in interval2 for x in interval1)


class TestCycleIntervalIntersection:

    @valid_type_test(CycleInterval(0, 0).intersection)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(CycleInterval(0, 0).intersection)
    def test_invalid_type(self) -> None:
        pass

    @given(
        tuples(cycle_intervals(), cycle_intervals())
        .filter(lambda x: x[0] < x[1]),
    )
    def test_invalid_value(
        self,
        interval: tuple[CycleInterval, CycleInterval],
    ) -> None:
        with pytest.raises(ValueError):
            interval[0].intersection(interval[1])

    @given(
        tuples(cycle_intervals(), cycle_intervals())
        .filter(lambda x: x[0].overlaps(x[1])),
    )
    def test_valid(
        self,
        interval: tuple[CycleInterval, CycleInterval],
    ) -> None:
        inter = interval[0].intersection(interval[1])
        assert isinstance(inter, CycleInterval)
        assert all(x in interval[0] and x in interval[1] for x in inter)
        inter = interval[1].intersection(interval[0])
        assert isinstance(inter, CycleInterval)
        assert all(x in interval[0] and x in interval[1] for x in inter)


class TestCycleIntervalUnion:

    @valid_type_test(CycleInterval(0, 0).union)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(CycleInterval(0, 0).union)
    def test_invalid_type(self) -> None:
        pass

    @given(
        tuples(cycle_intervals(), cycle_intervals())
        .filter(lambda x: x[0] < x[1] and x[0].upper < x[1].lower - 1),
    )
    def test_invalid(
        self,
        interval: tuple[CycleInterval, CycleInterval],
    ) -> None:
        with pytest.raises(ValueError):
            interval[0].union(interval[1])

    @given(
        tuples(cycle_intervals(), cycle_intervals())
        .filter(lambda x: x[0].overlaps(x[1])),
    )
    def test_valid(
        self,
        interval: tuple[CycleInterval, CycleInterval],
    ) -> None:
        union = interval[0].union(interval[1])
        assert isinstance(union, CycleInterval)
        assert all(x in interval[0] or x in interval[1] for x in union)
        union = interval[1].union(interval[0])
        assert isinstance(union, CycleInterval)
        assert all(x in interval[0] or x in interval[1] for x in union)


class TestCycleIntervalLt:

    @given(cycle_intervals(), everything_except(tuple))
    # @example(CycleInterval(0, 0), tuple())
    def test_invalid(self, interval: CycleInterval, other: Any) -> None:
        with pytest.raises(TypeError):
            interval < other
        with pytest.raises(TypeError):
            other > interval

    @given(cycle_intervals(), cycle_intervals())
    def test_valid(
        self,
        interval1: CycleInterval,
        interval2: CycleInterval,
    ) -> None:
        if interval1.overlaps(interval2):
            assert not(interval1 < interval2 or interval2 < interval1)
        else:
            assert interval1 < interval2 or interval2 < interval1
            if interval1 < interval2:
                assert all(x < interval2.lower for x in interval1)
            else:
                assert all(x < interval1.lower for x in interval2)


class TestCycleIntervalIsBounds:

    @given(cycle_intervals())
    def test_true_1(self, interval: CycleInterval) -> None:
        assert CycleInterval.is_interval(interval)

    @given(tuples(integers(), integers()))
    def test_true_2(self, interval: tuple[int, int]) -> None:
        assert CycleInterval.is_interval(interval)

    @given(type_annotation_to_invalid_strategy('Tuple[int, int]'))
    def test_false(self, not_interval: Any) -> None:
        assert not CycleInterval.is_interval(not_interval)
