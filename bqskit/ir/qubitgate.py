from typing import List

from bqskit.ir.gate import Gate


class QubitGate(Gate):

    def get_radix(self) -> List[int]:
        return [2] * self.get_gate_size()
