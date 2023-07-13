"""This module implements the SwapGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.qis.permutation import PermutationMatrix
from bqskit.utils.typing import is_integer

class SwapGate(ConstantGate):
    """
    The two-qudit swap gate.

    This gate swaps the state of two qudits. For example, The qubit swap
    gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        \\end{pmatrix}
    """

    def __init__(self, radix: int = 2) -> None:
        """
        Create a swap gate, defaulting to the qubit swap gate.

        Args:
            radix (int): The base of the qudits being swapped.
                Defaults to qubits or base 2. (Default: 2)

        Raises:
            ValueError: If radix is less than two.
        """
        if not is_integer(radix):
            raise TypeError('Expected a single integer radix.')

        if radix < 2:
            raise ValueError('Radix must be at least 2.')

        self._num_qudits = 2
        self._radixes = (radix, radix)
        self._dim = radix * radix
        self._utry = PermutationMatrix.gen_swap_unitary(radix)
        self._qasm_name = 'swap'

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SwapGate)
            and self.radixes == other.radixes
        )

    def __hash__(self) -> int:
        return hash(('swapgate', self.radixes[0]))

    def __str__(self) -> str:
        if self.is_qubit_only():
            return 'SwapGate'
        else:
            return f'SwapGate({self.radixes[0]})'
