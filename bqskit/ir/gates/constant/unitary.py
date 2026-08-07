"""This module implements the ConstantUnitaryGate."""
from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from openqudit.expressions import UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.qis import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


def _qgl_num(x: float) -> str:
    """Render a float as a QGL constant (no exponent, unary '~' for negation)"""
    if x == 0.0:
        return '0'
    s = format(Decimal(x), 'f')          # exact, never scientific notation
    return f'~{s[1:]}' if s.startswith('-') else s


def _qgl_entry(z: complex) -> str:
    re, im = _qgl_num(z.real), _qgl_num(z.imag)
    if im == '0':
        return re
    if re == '0':
        return f'{im}*i'
    return f'{re} + {im}*i'


def _utry_to_qgl(utry: UnitaryMatrix, name: str) -> str:
    m = utry.numpy
    rows = ',\n    '.join(
        '[' + ', '.join(_qgl_entry(z) for z in row) + ']'
        for row in m
    )
    radices = ', '.join(str(r) for r in utry.radixes)
    return f'{name}<{radices}>() {{\n  [{rows}]\n}}'


class ConstantUnitaryGate(Gate, CachedClass):
    """An arbitrary constant unitary operator."""

    _num_params = 0

    def __init__(
        self,
        utry: UnitaryLike,
        radixes: Sequence[int] = [],
    ) -> None:
        """
        Construct a constant unitary operator.

        Args:
            utry (UnitaryLike): The operation as a unitary matrix.

            radixes (Sequence[int]): The number of orthogonal states
                for each qudit this gate will act on. Defaults to qubits.
        """
        self._utry = UnitaryMatrix(utry, radixes)
        name = f'ConstUtry_{abs(hash(self._utry)):x}'
        self._expr = UnitaryExpression(_utry_to_qgl(self._utry, name))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ConstantUnitaryGate)
            and self._utry == other._utry
        )

    def __hash__(self) -> int:
        return hash(self._utry)
