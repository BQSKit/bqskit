"""This module tests the RSU3Gate class."""
from __future__ import annotations

import numpy as np
import scipy as sp

from bqskit.ir.gates import RSU3Gate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.math import compute_su_generators


def test_get_unitary() -> None:
    generators = compute_su_generators(3)
    angle = np.random.random()

    for i in range(8):
        g = RSU3Gate(i)
        u = UnitaryMatrix(sp.linalg.expm(-1j * angle * generators[i]), [3])
        assert g.get_unitary([angle]).get_distance_from(u) < 1e-7
