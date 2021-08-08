"""This module tests the CycleInterval class."""
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis.strategies import dictionaries
from hypothesis.strategies import integers

from bqskit.ir.region import CircuitRegion
from bqskit.test.strategy import cycle_intervals
from bqskit.test.types import invalid_type_test
from bqskit.test.types import type_annotation_to_invalid_strategy
from bqskit.test.types import valid_type_test


class TestCircuitRegionInit:

    @valid_type_test(CircuitRegion)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(CircuitRegion)
    def test_invalid_type(self) -> None:
        pass

    # @given(dictionaries(integers(0), cycle_intervals()))
    # def
