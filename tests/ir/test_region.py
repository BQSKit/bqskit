"""This module tests the CycleInterval class."""
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis.strategies import dictionaries
from hypothesis.strategies import integers

from bqskit.ir.interval import CycleInterval
from bqskit.ir.region import CircuitRegion
from bqskit.test.strategy import circuit_regions
from bqskit.test.strategy import cycle_intervals
from bqskit.test.types import invalid_type_test
from bqskit.test.types import valid_type_test


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
