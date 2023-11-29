"""This module implements the ControlledGate class."""
from __future__ import annotations

from typing import cast
from typing import Sequence

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.docs import building_docs
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_sequence
from bqskit.utils.typing import is_sequence_of_int
from bqskit.utils.typing import is_valid_radixes


class EmbeddedGate(ComposedGate, DifferentiableUnitary):
    """
    An embedding of a gate into a higher-dimensional qudit gate.

    For example, a qubit gate can be embedded into a qutrit gate by
    mapping all qubit levels to a subspace of the qutrit levels.

    This transformation can be shown directly on unitaries. If we have an
    arbitrary single-qubit gate :math:`U` given by the following matrix:

    .. math::

        U = \\begin{pmatrix}
            a & b \\\\
            c & d \\\\
        \\end{pmatrix}

    and we want to embed this into the 0 and 2 levels of a qutrit gate,
    then we can do so by mapping the qubit levels 0 and 1 to the qutrit
    levels 0 and 2, respectively. This gives us the following matrix:

    .. math::

        U_{embedded} = \\begin{pmatrix}
            a & 0 & b \\\\
            0 & 1 & 0 \\\\
            c & 0 & d \\\\
        \\end{pmatrix}

    This concept can be generalized to multiple qudits and even
    mixed-radix systems.

    Note:
        - Global phase inconsistencies in gates will become local phase
            inconsistencies in the embedded gate. For example, if the
            global phase difference between the U1Gate and the RZGate
            will become local phase differences in the corresponding
            subspaces when embedded into a higher-dimensional qudit.
    """

    def __init__(
        self,
        gate: Gate,
        radixes: Sequence[int] | int,
        level_maps: None | Sequence[int] | Sequence[Sequence[int]] = None,
    ) -> None:
        """
        Construct an EmbeddedGate.

        Args:
            gate (Gate): The gate to embed in a higher-dimensional qudit
                gate.

            radixes (Sequence[int] | int): The target radixes of the higher-
                dimensional system. If an integer is given, then the radixes
                are assumed to be the same for all qudits. For example,
                if `radixes = 3`, then the gate will be embedded into a
                qutrit gate.

             level_maps (None | Sequence[int] | Sequence[Sequence[int]]):
                 The level map for the embedding for each qudit. If a
                 sequence of integers is given, then the level map is
                 assumed to be the same for all qudits. For example, if
                 `radixes = 3` and `level_maps = [0, 2]`, then the gate
                 will be embedded into a qutrit gate by mapping the qubit
                 levels 0 and 1 to the qutrit levels 0 and 2,
                 respectively. If a sequence of sequences is given, then
                 the level map is assumed to be different for each qudit.
                 For example, if `radixes = [3, 3]` and `level_maps =
                 [[0, 2], [1, 2]]`, then the gate will be embedded into a
                 two-qudit gate by mapping the first qubit's 0 and 1
                 levels to the first qutrit's 0 and 2 levels,
                 respectively, and by mapping the second qubit's 0 and 1
                 levels to the second qutrit's 1 and 2 levels,
                 respectively. This can also be set to `None`, which will
                 embed the lower dimension gate in the lowest levels of
                 the new radixes.

        Raises:

            ValueError: If any radix is less than 2.

            ValueError: If radixes is given as a sequence and its length
                is not equal to the number of qudits in the gate.

            ValueError: If any of the gate's radixes are greater than the
                corresponding target radixes.

            ValueError: If the level map is given as a sequence of sequences
                and its length is not equal to the number of qudits in the
                gate.

            ValueError: If any of the individual qudit level maps are not
                the same length as the gate's corresponding qudit radix.

            ValueError: If any individual qudit level map has an invalid
                qudit level, i.e. too low (< 0) or too high (>= radix).

            ValueError: If any individual qudit level map is not one-to-one,
                i.e. if any two qudit levels are mapped to the same target
                qudit level.

        Examples: (#TODO update)
            XGate for qutrits:
                ```
                > x_qutrit_gate = ControlledGate(XGate(),[3],[0,2])
                > x_qutrit_gate.get_unitary()
                > [[0, 0, sqrt(2)/2],
                   [0, 0, 0],
                   [sqrt(2)/2, 0, 0]]
                ```
        """
        if not isinstance(gate, Gate):
            raise TypeError(f'Expected gate object, got {type(gate)}.')

        if is_integer(radixes):
            radixes = [radixes] * gate.num_qudits

        radixes = cast(Sequence[int], radixes)

        if not is_valid_radixes(radixes, gate.num_qudits):
            raise ValueError(
                'Given target radixes was not valid. Either invalid type,'
                ' invalid length, or invalid radix. Expected target radixes'
                ' to be a single integer or sequence of integers with length'
                f' equal to {gate.num_qudits=}. Also expected every radix'
                f' to be greater than 2, got {radixes=}.',
            )

        if level_maps is None:
            level_maps = [list(range(levels)) for levels in gate.radixes]

        if not is_sequence(level_maps):
            raise TypeError(
                'Expected level_maps to be a sequence of integers or a '
                f'sequence of sequences of integers, got {level_maps}.',
            )

        if is_sequence_of_int(level_maps):
            level_maps = [level_maps] * gate.num_qudits

        if not all(is_sequence_of_int(level_map) for level_map in level_maps):
            raise TypeError(
                'Expected level_maps to be a sequence of integers or a '
                f'sequence of sequences of integers, got {level_maps}.',
            )

        level_maps = cast(Sequence[Sequence[int]], level_maps)

        if any(gr > tr for gr, tr in zip(gate.radixes, radixes)):
            raise ValueError(
                'Given target radixes was not valid. Expected every target'
                ' radix to be greater than or equal to the corresponding'
                f' gate radix, got {gate.radixes=} and {radixes=}.',
            )

        if len(level_maps) != gate.num_qudits:
            raise ValueError(
                'Given level_maps was not valid. Expected level_maps to be'
                ' a sequence of sequences of ints with length equal to '
                f'{gate.num_qudits=}, got {len(level_maps)=}.',
            )

        if any(len(lmap) != gr for gr, lmap in zip(gate.radixes, level_maps)):
            raise ValueError(
                'Given level_maps was not valid. Expected every level_map'
                ' to have length equal to the corresponding gate radix, got'
                f' {gate.radixes=} and {level_maps=}.',
            )

        if any(
            any(lvl < 0 or lvl >= r for lvl in lmap)
            for r, lmap in zip(radixes, level_maps)
        ):
            raise ValueError(
                'Given level_maps was not valid. Expected every level_map'
                ' to have all levels in the range [0, radix), got'
                f' {radixes=} and {level_maps=}.',
            )

        if any(len(lmap) != len(set(lmap)) for lmap in level_maps):
            raise ValueError(
                'Given level_maps was not valid. Expected every level_map'
                f' to be one-to-one, got duplicate levels: {level_maps=}.',
            )

        self.gate = gate
        self.level_maps = tuple([tuple(list(lmap)) for lmap in level_maps])
        self._num_qudits = gate._num_qudits
        self._name = 'Embedded(%s)%s' % (self.gate.name, self.level_maps)
        self._num_params = self.gate._num_params
        self._radixes = tuple(radixes)
        self._dim = int(np.prod(self.radixes))

        # If input is a constant gate, we can cache the unitary.
        if self.num_params == 0 and not building_docs():
            U = self.gate.get_unitary()
            U_embed = np.eye(self.dim, dtype=np.complex128)
            self._map_matrix(U, U_embed)
            self._utry = UnitaryMatrix(U_embed, self.radixes, False)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        if hasattr(self, '_utry'):
            return self._utry

        U = self.gate.get_unitary(params)
        U_embed = np.eye(self.dim, dtype=np.complex128)
        self._map_matrix(U, U_embed)
        return UnitaryMatrix(U_embed, self.radixes, False)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if hasattr(self, 'utry'):
            return np.array([])

        G = self.gate.get_grad(params)  # type: ignore
        G_embed = []
        for g in G:
            M = np.zeros((self.dim, self.dim), dtype=np.complex128)
            self._map_matrix(g, M)
            G_embed.append(M)
        return np.array(G_embed, dtype=np.complex128)

    def get_unitary_and_grad(
        self,
        params: RealVector = [],
    ) -> tuple[UnitaryMatrix, npt.NDArray[np.complex128]]:
        """
        Return the unitary and gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if hasattr(self, '_utry'):
            return self._utry, np.array([])

        U, G = self.gate.get_unitary_and_grad(params)  # type: ignore
        U_embed = np.eye(self.dim, dtype=np.complex128)
        self._map_matrix(U, U_embed)

        G_embed = []
        for g in G:
            M = np.zeros((self.dim, self.dim), dtype=np.complex128)
            self._map_matrix(g, M)
            G_embed.append(M)

        return (
            UnitaryMatrix(U_embed, self.radixes, False),
            np.array(G_embed, dtype=np.complex128),
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, EmbeddedGate)
            and self.gate == other.gate
            and self.radixes == other.radixes
            and self.level_maps == other.level_maps
        )

    def __hash__(self) -> int:
        return hash((self.gate, self.radixes, self.level_maps))

    def _map_matrix(
        self,
        small: npt.NDArray[np.complex128] | UnitaryMatrix,
        big: npt.NDArray[np.complex128],
    ) -> None:
        """
        Map the lower-dimensional `small` matrix into the larger `big` matrix.

        This will mutate the `big` matrix in-place. The `small` matrix
        must be a square matrix with dimensions equal to the dimension
        of the gate being embedded. The `big` matrix must be a square
        matrix with dimensions equal to the dimension of the gate being
        embedded into.

        The embedding is done by mapping the indices of the `small`
        matrix to the indices of the `big` matrix using the level maps.

        When doing unitary calculations, pass the identity in for the
        `big` matrix and the unitary of the gate being embedded in for
        the `small` matrix.

        When doing gradient calculations, pass the zero matrix in for
        the `big` matrix and the gradient of the gate being embedded in
        for the `small` matrix.

        Args:
            small (npt.NDArray[np.complex128]): The matrix to embed.

            big (npt.NDArray[np.complex128]): The matrix to embed into.

        Notes:
            No checks are done to ensure that the parameters are correct.
        """

        for i in range(self.gate.dim):
            # Expand i, j into the mixed-radix basis of the gate
            i_exp = np.unravel_index(i, self.gate.radixes)

            # Map the indices to the target basis using the level maps
            i_map_exp = [lm[ie] for lm, ie in zip(self.level_maps, i_exp)]

            # Convert the mapped expanded indices back to a flat index
            i_target = np.ravel_multi_index(i_map_exp, self.radixes)

            for j in range(self.gate.dim):
                # Expand i, j into the mixed-radix basis of the gate
                j_exp = np.unravel_index(j, self.gate.radixes)

                # Map the indices to the target basis using the level maps
                j_map_exp = [lm[je] for lm, je in zip(self.level_maps, j_exp)]

                # Convert the mapped expanded indices back to a flat index
                j_target = np.ravel_multi_index(j_map_exp, self.radixes)
                big[i_target, j_target] = small[i, j]
