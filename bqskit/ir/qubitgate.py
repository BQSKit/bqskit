from __future__ import annotations

from bqskit.ir.gate import Gate


class QubitGate(Gate):

    def get_radix(self) -> list[int]:
        return [2] * self.get_gate_size()
