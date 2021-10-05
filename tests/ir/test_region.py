"""This module tests the CycleInterval class."""
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis.strategies import dictionaries
from hypothesis.strategies import integers
from hypothesis.strategies import tuples

from bqskit.ir.interval import CycleInterval
from bqskit.ir.location import CircuitLocation
from bqskit.ir.point import CircuitPoint
from bqskit.ir.region import CircuitRegion
from bqskit.test.strategies import circuit_points
from bqskit.test.strategies import circuit_regions
from bqskit.test.strategies import cycle_intervals
from bqskit.test.types import invalid_type_test
from bqskit.test.types import valid_type_test


@given(circuit_regions())
def test_strategy(region: CircuitRegion) -> None:
    assert isinstance(region, CircuitRegion)
    assert CircuitRegion.is_region(region)


class TestCircuitRegionInit:

    @valid_type_test(CircuitRegion)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(CircuitRegion)
    def test_invalid_type(self) -> None:
        pass

    @given(dictionaries(integers(0), cycle_intervals()))
    def test_valid(self, region_map: dict[int, CycleInterval]) -> None:
        region = CircuitRegion(region_map)
        assert all(x in region for x in region_map)
        assert len(region) == len(region_map)
        assert all(region[x] == region_map[x] for x in region_map)


@given(circuit_regions(), integers())
def test_get_item(region: CircuitRegion, key: int) -> None:
    if key in region:
        assert isinstance(region[key], CycleInterval)
    else:
        with pytest.raises(KeyError):
            region[key]


@given(circuit_regions())
def test_iter(region: CircuitRegion) -> None:
    for key in region:
        assert key in region
        assert isinstance(key, int)
        assert isinstance(region[key], CycleInterval)


@given(circuit_regions())
def test_len(region: CircuitRegion) -> None:
    assert len(region) == len(list(region))


@given(circuit_regions())
def test_min_cycle(region: CircuitRegion) -> None:
    if region.empty:
        with pytest.raises(ValueError):
            min_cycle = region.min_cycle
        return

    min_cycle = region.min_cycle
    assert all(min_cycle <= interval.lower for interval in region.values())
    assert any(min_cycle == interval.lower for interval in region.values())


@given(circuit_regions())
def test_max_cycle(region: CircuitRegion) -> None:
    if region.empty:
        with pytest.raises(ValueError):
            max_cycle = region.max_cycle
        return

    max_cycle = region.max_cycle
    assert all(max_cycle >= interval.upper for interval in region.values())
    assert any(max_cycle == interval.upper for interval in region.values())


@given(circuit_regions())
def test_max_min_cycle(region: CircuitRegion) -> None:
    if region.empty:
        with pytest.raises(ValueError):
            max_min_cycle = region.max_min_cycle
        return

    max_min_cycle = region.max_min_cycle
    assert all(max_min_cycle >= interval.lower for interval in region.values())
    assert any(max_min_cycle == interval.lower for interval in region.values())


@given(circuit_regions())
def test_min_max_cycle(region: CircuitRegion) -> None:
    if region.empty:
        with pytest.raises(ValueError):
            min_max_cycle = region.min_max_cycle
        return

    min_max_cycle = region.min_max_cycle
    assert all(min_max_cycle <= interval.upper for interval in region.values())
    assert any(min_max_cycle == interval.upper for interval in region.values())


@given(circuit_regions())
def test_min_qudit(region: CircuitRegion) -> None:
    if region.empty:
        with pytest.raises(ValueError):
            min_qudit = region.min_qudit
        return

    min_qudit = region.min_qudit
    assert all(min_qudit <= qudit for qudit in region.keys())
    assert any(min_qudit == qudit for qudit in region.keys())
    assert min(region) == region.min_qudit


@given(circuit_regions())
def test_max_qudit(region: CircuitRegion) -> None:
    if region.empty:
        with pytest.raises(ValueError):
            max_qudit = region.max_qudit
        return

    max_qudit = region.max_qudit
    assert all(max_qudit >= qudit for qudit in region.keys())
    assert any(max_qudit == qudit for qudit in region.keys())
    assert max(region) == region.max_qudit


@given(circuit_regions())
def test_location(region: CircuitRegion) -> None:
    assert isinstance(region.location, CircuitLocation)
    assert all(x in region.keys() for x in region.location)
    assert len(region.location) == len(region)
    assert all(q1 < q2 for q1, q2 in zip(region.location, region.location[1:]))


@given(circuit_regions())
def test_points(region: CircuitRegion) -> None:
    points = region.points
    assert isinstance(points, list)
    assert all(isinstance(point, CircuitPoint) for point in points)
    assert all(point in region for point in points)
    assert region.volume == len(points)


@given(circuit_regions())
def test_volume(region: CircuitRegion) -> None:
    volume = region.volume
    assert isinstance(volume, int)
    assert volume >= len(region.keys())
    assert volume <= len(region.keys()) * region.width
    assert region.volume == len(region.points)


@given(circuit_regions())
def test_width(region: CircuitRegion) -> None:
    width = region.width

    if region.empty:
        assert width == 0
        return

    assert isinstance(width, int)
    assert width <= region.max_cycle + 1
    assert all(width >= len(interval) for interval in region.values())

    pmin = min(point.cycle for point in region.points)
    pmax = max(point.cycle for point in region.points)
    assert width == pmax - pmin + 1


@given(circuit_regions())
def test_empty(region: CircuitRegion) -> None:
    assert region.empty == (len(region.keys()) == 0)


class TestShiftLeft:

    @valid_type_test(CircuitRegion({}).shift_left)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(CircuitRegion({}).shift_left)
    def test_invalid_type(self) -> None:
        pass

    @given(
        tuples(circuit_regions(empty=False), integers(0, 20))
        .filter(lambda x: 0 < x[1] <= x[0].min_cycle),
    )
    def test_valid(self, data: tuple[CircuitRegion, int]) -> None:
        region, amount_to_shift = data
        points_before_shift = region.points
        shifted_region = region.shift_left(amount_to_shift)
        points_after_shift = shifted_region.points
        assert all(
            (point[0] - amount_to_shift, point[1]) in points_after_shift
            for point in points_before_shift
        )
        assert region is not shifted_region

    @given(tuples(circuit_regions(empty=False), integers(-20, -1)))
    def test_negative(self, data: tuple[CircuitRegion, int]) -> None:
        region, amount_to_shift = data
        points_before_shift = region.points
        shifted_region = region.shift_left(amount_to_shift)
        points_after_shift = shifted_region.points
        assert all(
            (point[0] - amount_to_shift, point[1]) in points_after_shift
            for point in points_before_shift
        )
        assert region is not shifted_region

    @given(circuit_regions())
    def test_invalid(self, region: CircuitRegion) -> None:
        with pytest.raises(ValueError):
            region.shift_left(region.min_cycle + 1)


class TestShiftRight:

    @valid_type_test(CircuitRegion({}).shift_right)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(CircuitRegion({}).shift_right)
    def test_invalid_type(self) -> None:
        pass

    @given(tuples(circuit_regions(empty=False), integers(0, 20)))
    def test_valid(self, data: tuple[CircuitRegion, int]) -> None:
        region, amount_to_shift = data
        points_before_shift = region.points
        shifted_region = region.shift_right(amount_to_shift)
        points_after_shift = shifted_region.points
        assert all(
            (point[0] + amount_to_shift, point[1]) in points_after_shift
            for point in points_before_shift
        )
        assert region is not shifted_region

    @given(
        tuples(circuit_regions(empty=False), integers(-20, -1))
        .filter(lambda x: 0 < -x[1] <= x[0].min_cycle),
    )
    def test_negative(self, data: tuple[CircuitRegion, int]) -> None:
        region, amount_to_shift = data
        points_before_shift = region.points
        shifted_region = region.shift_right(amount_to_shift)
        points_after_shift = shifted_region.points
        assert all(
            (point[0] + amount_to_shift, point[1]) in points_after_shift
            for point in points_before_shift
        )
        assert region is not shifted_region

    @given(circuit_regions())
    def test_invalid(self, region: CircuitRegion) -> None:
        with pytest.raises(ValueError):
            region.shift_right(-region.min_cycle - 1)


class TestOverlaps:

    @valid_type_test(CircuitRegion({}).overlaps)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(CircuitRegion({}).overlaps)
    def test_invalid_type(self) -> None:
        pass

    @given(circuit_regions(), circuit_regions())
    def test_region_by_region(
        self,
        region1: CircuitRegion,
        region2: CircuitRegion,
    ) -> None:
        if region1.overlaps(region2):
            assert any(x in region2.points for x in region1.points)
        else:
            assert all(x not in region2.points for x in region1.points)

    @given(circuit_points(), circuit_regions())
    def test_region_by_point(
        self,
        point: CircuitPoint,
        region: CircuitRegion,
    ) -> None:
        if region.overlaps(point):
            assert point in region.points
        else:
            assert point not in region.points


class TestContains:

    @given(circuit_regions(), circuit_regions())
    def test_region_by_region(
        self,
        region1: CircuitRegion,
        region2: CircuitRegion,
    ) -> None:
        if region1 in region2:
            assert region2 not in region1 or region1 == region2
            assert all(x in region2.points for x in region1.points)
        elif region2 in region1:
            assert region1 not in region2 or region1 == region2
            assert all(x in region1.points for x in region2.points)
        else:
            assert region1 != region2
            assert any(x not in region2.points for x in region1.points)
            assert any(x not in region1.points for x in region2.points)

    @given(circuit_points(), circuit_regions())
    def test_region_by_point(
        self,
        point: CircuitPoint,
        region: CircuitRegion,
    ) -> None:
        if point in region:
            assert point in region.points
        else:
            assert point not in region.points

    @given(integers(0), circuit_regions())
    def test_region_by_int(
        self,
        integer: int,
        region: CircuitRegion,
    ) -> None:
        if integer in region:
            assert integer in region.keys()
        else:
            assert integer not in region.keys()


@given(circuit_regions())
def test_transpose(region: CircuitRegion) -> None:
    transpose = region.transpose()
    assert isinstance(transpose, dict)
    assert all(isinstance(k, int) for k in transpose.keys())
    assert all(isinstance(v, list) for v in transpose.values())
    assert all(isinstance(q, int) for v in transpose.values() for q in v)
    for k, v in transpose.items():
        for q in v:
            assert (k, q) in region.points


class TestIntersection:

    @valid_type_test(CircuitRegion({}).intersection)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(CircuitRegion({}).intersection)
    def test_invalid_type(self) -> None:
        pass

    @given(circuit_regions(), circuit_regions())
    def test_region_by_region(
        self,
        region1: CircuitRegion,
        region2: CircuitRegion,
    ) -> None:
        inter = region1.intersection(region2)
        assert isinstance(inter, CircuitRegion)
        assert all(x in region1 and x in region2 for x in inter.points)
        inter = region2.intersection(region1)
        assert isinstance(inter, CircuitRegion)
        assert all(x in region1 and x in region2 for x in inter.points)


class TestUnion:

    @valid_type_test(CircuitRegion({}).union)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(CircuitRegion({}).union)
    def test_invalid_type(self) -> None:
        pass

    @given(circuit_regions(), circuit_regions())
    def test_region_by_region(
        self,
        region1: CircuitRegion,
        region2: CircuitRegion,
    ) -> None:
        for qudit in region1:
            if qudit in region2:
                if region1[qudit].overlaps(region2[qudit]):
                    continue
                elif (
                    region1[qudit].upper + 1 != region2[qudit].lower
                    and region1[qudit].lower - 1 != region2[qudit].upper
                ):
                    with pytest.raises(ValueError):
                        region1.union(region2)
                    return

        union = region1.union(region2)
        assert isinstance(union, CircuitRegion)
        assert all(x in region1 or x in region2 for x in union.points)
        union = region2.union(region1)
        assert isinstance(union, CircuitRegion)
        assert all(x in region1 or x in region2 for x in union.points)


@given(circuit_regions(), circuit_regions())
def test_depends_on(
    region1: CircuitRegion,
    region2: CircuitRegion,
) -> None:
    qudits = region1.location.intersection(region2.location)

    r1_depends_on_r2 = region1.depends_on(region2)
    r2_depends_on_r1 = region2.depends_on(region1)
    if len(qudits) == 0:
        assert not r1_depends_on_r2
        assert not r2_depends_on_r1

    else:
        if r1_depends_on_r2 and r2_depends_on_r1:
            assert any(
                qudit in region1 and region2[qudit] < region1[qudit]
                for qudit in region2
            )
            assert any(
                qudit in region2 and region1[qudit] < region2[qudit]
                for qudit in region1
            )
        elif r1_depends_on_r2:
            assert all(
                qudit not in region1 or region2[qudit] <= region1[qudit]
                for qudit in region2
            ) and any(
                region2[qudit] < region1[qudit]
                for qudit in region2
            )
        elif r2_depends_on_r1:
            assert all(
                qudit not in region2 or region1[qudit] <= region2[qudit]
                for qudit in region1
            ) and any(
                region1[qudit] < region2[qudit]
                for qudit in region1
            )
