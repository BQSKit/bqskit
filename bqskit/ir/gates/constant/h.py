"""This module implements the HGate."""
from __future__ import annotations

from openqudit.expressions import HGate as _HGate

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.quditgate import QuditGate
from bqskit.utils.cachedclass import CachedClass
from bqskit.utils.typing import is_integer


class HGate(ConstantGate, QuditGate, CachedClass):
    """
    The one-qudit Hadamard gate. This is a Clifford gate.

    By default, the HGate is a qubit Hadamard gate. However, it can be
    generalized to qudits by setting the radix parameter during construction.

    It is represented by the following matrix for qubits:

    .. math::

        \\begin{pmatrix}
        \\frac{\\sqrt{2}}{2} & \\frac{\\sqrt{2}}{2} \\\\
        \\frac{\\sqrt{2}}{2} & -\\frac{\\sqrt{2}}{2} \\\\
        \\end{pmatrix}

    However, generally it is given by the following formula:

    .. math::
        H = \\frac{1}{\\sqrt(d)} \\sum_{ij} \\omega_d^{ij} |i\\rangle\\langle j|

    where

    .. math::
        \\omega = \\exp(\\frac{2\\pi i}{d})

    and `d` is the number of levels (2 levels is a qubit,
    3 levels is a qutrit, etc.)

    References:
        - https://www.frontiersin.org/articles/10.3389/fphy.2020.589504/full
        - https://pubs.aip.org/aip/jmp/article-abstract/56/3/032202/763827
    """

    _num_qudits = 1
    _num_params = 0
    _qasm_name = 'h'

    def __init__(self, radix: int = 2) -> None:
        """
        Construct a HGate.

        Args:
            radix (int): The number of qudit levels (>=2). By default, this
                is 2, which is gives a qubit Hadamard gate.

        Raises:
            ValueError: if radix < 2
        """
        if not is_integer(radix):
            raise TypeError(f"Expected integer for radix, got {type(radix)}.")

        if radix < 2:
            raise ValueError(f"Radix must be greater than 1, got {radix}.")

        self._radix = radix
        self._expr = _HGate(radix)
