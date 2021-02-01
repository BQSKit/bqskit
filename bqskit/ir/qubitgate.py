from typing import Iterable

from bqskit.ir.gate import Gate


class QubitGate(Gate):

    def get_radix(self) -> Iterable[int]:
        return [2] * self.get_gate_size()
