from typing import Sequence
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix, UnitaryLike
from bqskit.utils.docs import building_docs

from bqskit.utils.typing import is_square_matrix

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla


class UnitaryMatrixJax(UnitaryMatrix):

    def __init__(
        self,
        input: UnitaryLike,
        radixes: Sequence[int] = [],
        _from_tree: bool = False,
        ) -> None:


        # Stop any actual logic when building documentation
        if building_docs():
            self._utry = jnp.array([])
            return
            
        self._my_class = UnitaryMatrixJax
        self._mat_lib = jnp
        # Copy constructor
        if isinstance(input, (UnitaryMatrixJax, UnitaryMatrix)):
            self._utry = jnp.array(input.numpy)
            self._radixes = input.radixes
            self._dim = input.dim
            return

        self._radixes = tuple(radixes)

        if type(input) is not object and type(input) is not jax.core.ShapedArray and not _from_tree:
                    self._utry = jnp.array(input, dtype=jnp.complex128).reshape(self.radixes * 2) # make sure its a square matrix
        else:
            self._utry = input

    @staticmethod
    def identity(dim: int, radixes: Sequence[int] = []):
        """
        Construct an identity UnitaryMatrix.

        Args:
            dim (int): The dimension of the identity matrix.

            radixes (Sequence[int]): The number of orthogonal states
                for each qudit, defaults to qubits.

        Returns:
            UnitaryMatrix: An identity matrix.

        Raises:
            ValueError: If `dim` is nonpositive.
        """
        if dim <= 0:
            raise ValueError('Invalid dimension for identity matrix.')
        return UnitaryMatrixJax(jnp.identity(dim), radixes)


    @staticmethod
    def closest_to(
        M,
        radixes: Sequence[int] = [],
    ):
        """
        Calculate and return the closest unitary to a given matrix.

        Calculate the unitary matrix U that is closest with respect to the
        operator norm distance to the general matrix M.

        Args:
            M (np.ndarray): The matrix input.

            radixes (Sequence[int]): The radixes for the Unitary.

        Returns:
            (UnitaryMatrix): The unitary matrix closest to M.

        References:
            D.M.Reich. “Characterisation and Identification of Unitary Dynamics
            Maps in Terms of Their Action on Density Matrices”
        """
        if not is_square_matrix(M):
            raise TypeError('Expected square matrix.')

        V, _, Wh = jla.svd(M)
        
        return UnitaryMatrixJax(V @ Wh, radixes)


    @staticmethod
    def random(num_qudits: int, radixes: Sequence[int] = []):
        return UnitaryMatrixJax(UnitaryMatrix.random(num_qudits, radixes))

    @staticmethod
    def from_file(filename: str):
        """Load a unitary from a file."""
        return UnitaryMatrixJax(jnp.loadtxt(filename, dtype=jnp.complex128))


    @property
    def T(self):
        """The transpose of the unitary."""
        return UnitaryMatrixJax(self._utry.T, self.radixes, False)

    def __array__(
        self,
        dtype = jnp.complex128,
    ):
        """Implements NumPy API for the UnitaryMatrix class."""
        if dtype != jnp.complex128:
            raise ValueError('UnitaryMatrix only supports JAX Complex128 dtype.')

        return self._utry



    def _tree_flatten(self):
        children = (self._utry,)  # arrays / dynamic values
        aux_data = {
            'radixes': self._radixes,
            '_from_tree': True,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


jax.tree_util.register_pytree_node(
    UnitaryMatrixJax,
    UnitaryMatrixJax._tree_flatten,
    UnitaryMatrixJax._tree_unflatten,
)
