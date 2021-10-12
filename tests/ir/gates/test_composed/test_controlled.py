"""This module tests the ControlledGate class."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates import CHGate
from bqskit.ir.gates import ControlledGate
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import SqrtCNOTGate
from bqskit.ir.gates import SqrtXGate
from bqskit.ir.gates import XGate


def test_ch() -> None:
    ch = ControlledGate(HGate())
    assert ch.get_unitary() == CHGate().get_unitary()


def test_cx() -> None:
    cx = ControlledGate(XGate())
    assert cx.get_unitary() == CXGate().get_unitary()


def test_csx() -> None:
    csx = ControlledGate(SqrtXGate())
    assert csx.get_unitary() == SqrtCNOTGate().get_unitary()


def test_toffoli() -> None:
    toffoli = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ])
    ccx = ControlledGate(XGate(), 2)
    assert ccx.get_unitary() == toffoli
