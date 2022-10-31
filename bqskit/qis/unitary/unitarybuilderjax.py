"""This module implements the UnitaryBuilder class."""
from __future__ import annotations

import logging
from typing import Sequence

import jax
import jax.numpy as jnp

from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitary import Unitary
from bqskit.qis.unitary.unitarybuilder import UnitaryBuilder
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.qis.unitary.unitarymatrixjax import UnitaryMatrixJax

logger = logging.getLogger(__name__)


class UnitaryBuilderJax(UnitaryBuilder):
    """
    An object for fast unitary accumulation using tensor networks.

    A UnitaryBuilder is similar to a StringBuilder in the sense that it is an
    efficient way to string together or accumulate :class:`Unitary` objects.
    This class uses concepts from tensor networks to efficiently multiply
    unitary matrices.
    """

    def __init__(self, num_qudits: int, radixes: Sequence[int] = [], initial_value: UnitaryMatrix = None) -> None:
        """
        UnitaryBuilder constructor.

        Args:
            num_qudits (int): The number of qudits to build a Unitary for.

            radixes (Sequence[int]): A sequence with its length equal
                to `num_qudits`. Each element specifies the base of a
                qudit. Defaults to qubits.

        Raises:
            ValueError: If `num_qudits` is nonpositive.

            ValueError: If the length of `radixes` is not equal to
                `num_qudits`.

        Examples:
            >>> builder = UnitaryBuilder(4)  # Creates a 4-qubit builder.
        """
        super().__init__(num_qudits, radixes, initial_value)

        self._mat_lib = jnp

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrixJax:
        """Build the unitary, see :func:`Unitary.get_unitary` for more."""
        utry = self.tensor.reshape((self.dim, self.dim))
        return UnitaryMatrixJax(utry, self.radixes)

    def _tree_flatten(self):
        children = (self.get_unitary(),)  # arrays / dynamic values
        aux_data = {
            'radixes': self._radixes,
            'num_qudits': self.num_qudits,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(initial_value=children[0], **aux_data)


jax.tree_util.register_pytree_node(
    UnitaryBuilderJax,
    UnitaryBuilderJax._tree_flatten,
    UnitaryBuilderJax._tree_unflatten,
)
