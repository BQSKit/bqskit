"""This module tests the StateVector class."""
from __future__ import annotations

import numpy as np
import pytest

from bqskit.qis.state.state import StateVector


class TestNew:

    def test_valid(self) -> None:
        state = StateVector([1, 0])
        assert isinstance(state, StateVector)
        assert isinstance(state, np.ndarray)
        assert state.dim == 2
        assert state.num_qudits == 1
        assert state.radixes == (2,)
        assert state.get_probs() == (1.0, 0.0)

    def test_copy(self) -> None:
        state1 = StateVector([1, 0])
        state2 = StateVector(state1)
        assert state1 is state2
        assert np.allclose(state1, state2)
        assert isinstance(state1, StateVector)
        assert isinstance(state1, np.ndarray)
        assert isinstance(state2, StateVector)
        assert isinstance(state2, np.ndarray)

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            StateVector([1, 1])


class TestNumpy:

    def test_basic(self) -> None:
        state1 = StateVector([1, 0])
        state2 = StateVector([0, 1])
        sum = state1 + state2
        assert isinstance(sum, np.ndarray)
        assert not isinstance(sum, StateVector)
