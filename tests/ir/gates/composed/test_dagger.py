"""This module tests the DaggerGate class."""
from __future__ import annotations

from bqskit.ir.gates import DaggerGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import TdgGate
from bqskit.ir.gates import TGate


def test_h() -> None:
    hdg = DaggerGate(HGate())
    assert hdg.get_unitary() == HGate().get_unitary()


def test_t() -> None:
    tdg = DaggerGate(TGate())
    assert tdg.get_unitary() == TdgGate().get_unitary()
