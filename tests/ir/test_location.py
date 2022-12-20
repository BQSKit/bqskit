"""This module tests the CircuitLocation class."""
from __future__ import annotations

from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.strategies import lists

from bqskit.ir.location import CircuitLocation
from bqskit.ir.location import CircuitLocationLike
from bqskit.utils.test.strategies import circuit_location_likes
from bqskit.utils.test.strategies import circuit_locations
from bqskit.utils.test.types import invalid_type_test
from bqskit.utils.test.types import valid_type_test


class TestInit:
    @invalid_type_test(CircuitLocation)
    def test_invalid_type(self) -> None:
        pass

    @valid_type_test(CircuitLocation)
    def test_valid_type(self) -> None:
        pass

    @given(lists(integers(0, 10), unique=True))
    def test_from_list(self, loc: list[int]) -> None:
        location = CircuitLocation(loc)
        assert len(location) == len(loc)
        assert all(x == y for x, y in zip(loc, location))


class TestUnion:
    @given(circuit_locations())
    def test_self(self, loc: CircuitLocation) -> None:
        assert set(loc.union(loc)) == set(loc)

    @given(circuit_locations(), circuit_location_likes())
    def test_other(self, l1: CircuitLocation, l2: CircuitLocationLike) -> None:
        union = l1.union(l2)
        assert len(union) >= len(l1)
        assert len(union) >= len(CircuitLocation(l2))
        assert all(x in union for x in l1)
        assert all(x in union for x in CircuitLocation(l2))


class TestIntersection:
    @given(circuit_locations())
    def test_self(self, loc: CircuitLocation) -> None:
        assert loc.intersection(loc) == loc

    @given(circuit_locations(), circuit_location_likes())
    def test_other(self, l1: CircuitLocation, l2: CircuitLocationLike) -> None:
        intersection = l1.intersection(l2)
        assert len(intersection) <= len(l1)
        assert len(intersection) <= len(CircuitLocation(l2))
        assert all(x in l1 for x in intersection)
        assert all(x in CircuitLocation(l2) for x in intersection)
