"""This module implements the SetTargetPass class."""
from __future__ import annotations

from typing import TYPE_CHECKING

from bqskit.compiler.basepass import BasePass
from bqskit.qis.state import StateSystem
from bqskit.qis.state import StateVector
from bqskit.qis.unitary import UnitaryMatrix

if TYPE_CHECKING:
    from bqskit.compiler.passdata import PassData
    from bqskit.ir.circuit import Circuit


class SetTargetPass(BasePass):
    """Sets a synthesis target for future passes."""

    def __init__(
        self,
        target: UnitaryMatrix | StateVector | StateSystem,
    ) -> None:
        """
        Construct a SetTargetPass.

        Args:
            target (UnitaryMatrix | StateVector | StateSystem): The target
                to synthesize in future passes.
        """
        if not isinstance(target, (StateVector, UnitaryMatrix, StateSystem)):
            bad_type = type(target)
            m = f'Expected valid unitary or state for target, got {bad_type}.'
            raise TypeError(m)

        self.target = target

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        data.target = self.target
