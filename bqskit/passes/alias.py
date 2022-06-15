"""This module implements the PassAlias class."""
from __future__ import annotations

import abc
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit


class PassAlias(BasePass):
    """A pass that is an alias for another pass or sequence of passes."""

    @abc.abstractmethod
    def get_passes(self) -> list[BasePass]:
        """Return the passes that should be run, when this alias is called."""

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        for bqskit_pass in self.get_passes():
            bqskit_pass.run(circuit, data)
