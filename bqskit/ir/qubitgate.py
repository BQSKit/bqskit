from __future__ import annotations

from bqskit.ir.gate import Gate


class QubitGate(Gate):

    @property
    def radixes(self) -> list[int]:
        """Returns the number of orthogonal states for each qudit."""
        return [2] * self.size
