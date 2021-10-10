"""This module tests the IdentityGate class."""
from __future__ import annotations

import numpy as np
from hypothesis import given

from bqskit.ir.gates import IdentityGate
from bqskit.test.strategies import num_qudits_and_radixes


@given(num_qudits_and_radixes())
def test_identity(pair: tuple[int, tuple[int, ...]]) -> None:
    num_qudits, radixes = pair
    i = IdentityGate(num_qudits, radixes)
    assert i.num_qudits == num_qudits
    assert i.num_params == 0
    assert i.get_unitary() == np.identity(int(np.prod(radixes)))
