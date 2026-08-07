"""This module implements the PauliZGate."""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gates.generalgate import GeneralGate
from bqskit.qis.pauliz import PauliZMatrices
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.docs import building_docs
from bqskit.utils.math import dot_product
from bqskit.utils.math import pauliz_expansion
from bqskit.utils.math import unitary_log_no_i


class PauliZGate(GeneralGate):
    """
    A gate representing an arbitrary diagonal rotation.

    This gate is given by:

    .. math::

        \\exp({i(\\vec{\\alpha} \\cdot \\vec{\\sigma_Z^{\\otimes n}})})

    Where :math:`\\vec{\\alpha}` are the gate's parameters,
    :math:`\\vec{\\sigma}` are the PauliZ Z matrices,
    and :math:`n` is the number of qubits this gate acts on.
    """

    def __init__(self, num_qudits: int) -> None:
        """
        Create a PauliZGate acting on `num_qudits` qubits.

        Args:
            num_qudits (int): The number of qudits this gate will act on.

        Raises:
            ValueError: If `num_qudits` is nonpositive.
        """
        if num_qudits <= 0:
            raise ValueError(f'Expected positive integer, got {num_qudits}')

        self._name = f'PauliZGate({num_qudits})'
        self._num_qudits = num_qudits
        self._radixes = tuple([2] * num_qudits)
        paulizs = PauliZMatrices(self.num_qudits)
        self._num_params = len(paulizs)
        if building_docs():
            self.sigmav: npt.NDArray[Any] = np.array([])
        else:
            self.sigmav = (-1j / 2) * paulizs.numpy

        dim = 2 ** num_qudits
        rows = [['0'] * dim for _ in range(dim)]
        for d in range(dim):
            terms = []
            for k in range(self._num_params):
                sign = int(round(paulizs.numpy[k][d, d].real))
                terms.append('t%d' % k if sign > 0 else '~t%d' % k)
            rows[d][d] = 'e^(~i*(%s)/2)' % '+'.join(terms)
        row_strs = ['[' + ','.join(row) + ']' for row in rows]
        params_str = ','.join('t%d' % k for k in range(self._num_params))
        self._expr = _UnitaryExpression(
            'PauliZ%d<%s>(%s) { [%s] }' % (
                num_qudits,
                ','.join(['2'] * num_qudits),
                params_str,
                ','.join(row_strs),
            ),
        )

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        H = dot_product(params, self.sigmav)
        eiH = np.diag(np.exp(np.diag(H)))
        return UnitaryMatrix(eiH, check_arguments=False)

    def calc_params(self, utry: UnitaryMatrix) -> list[float]:
        """Return the parameters for this gate to implement `utry`"""
        return list(-2 * pauliz_expansion(unitary_log_no_i(utry.numpy)))

    def __eq__(self, o: object) -> bool:
        return isinstance(o, PauliZGate) and self.num_qudits == o.num_qudits

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.num_qudits))
