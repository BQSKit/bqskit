"""This module tests the EmbeddedGate class."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates import HGate
from bqskit.ir.gates import XGate
from bqskit.ir.gates import ShiftGate
from bqskit.ir.gates import EmbeddedGate

def test_embded_h() -> None:
    EH =np.array([
        [1.0/np.sqrt(2), 1.0/np.sqrt(2), 0],
        [1.0/np.sqrt(2), -1.0/np.sqrt(2), 0],
        [0, 0, 1],
    ]) 
    eh = EmbeddedGate(HGate(),3,[0,1])
    assert eh.get_unitary() == EH


def test_embded_x() -> None:
    EX =np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ]) 
    ex = EmbeddedGate(XGate(),3,[0,2])
    assert ex.get_unitary() == EX


def test_embded_shift() -> None:
    ES =np.array([
        [0, 0, 1 ,0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]) 
    es = EmbeddedGate(ShiftGate(),4,[0,2,3])
    assert es.get_unitary() == ES
