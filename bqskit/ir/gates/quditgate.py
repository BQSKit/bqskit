"""This module implements the QuditGate base class."""
from __future__ import annotations
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

from bqskit.ir.gate import Gate


class QuditGate(Gate):
    """A gate that acts on qudits."""

    _num_levels: int

    @property
    def radixes(self) -> tuple[int, ...]:
        """The number of orthogonal states for each qudit."""
        return tuple([self.num_levels] * self.num_qudits)

    @property
    def num_levels(self) -> int:
        """The number of levels in each qudit."""
        return getattr(self, '_num_levels')
    
    #def build_single_qudit_unitary(self, qubitUnitary: UnitaryMatrix) -> UnitaryMatrix: #TODO fix
    #    assert hasattr(self, "level_1")
    #    assert hasattr(self, "level_0") 
    #    result =np.eye()
