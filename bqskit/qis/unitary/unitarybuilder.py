"""This module implements the UnitaryBuilder class."""
from __future__ import annotations

import logging
from typing import cast
from typing import Sequence

import numpy as np
import jax.numpy as jnp
import jax
import numpy.typing as npt


from bqskit.ir.location import CircuitLocation
from bqskit.ir.location import CircuitLocationLike
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitary import Unitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_valid_radixes

logger = logging.getLogger(__name__)


class UnitaryBuilder(Unitary):
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

        if not is_integer(num_qudits):
            raise TypeError(
                'Expected int for num_qudits, got %s.' %
                type(num_qudits),
            )

        if num_qudits <= 0:
            raise ValueError(
                'Expected positive number for num_qudits, got %d.' %
                num_qudits,
            )

        self._num_qudits = num_qudits
        self._radixes = tuple(radixes if len(radixes) > 0 else [2] * num_qudits)

        if not is_valid_radixes(self.radixes):
            raise TypeError('Invalid qudit radixes.')

        if len(self.radixes) != self.num_qudits:
            raise ValueError(
                'Expected length of radixes to be equal to num_qudits:'
                ' %d != %d' % (len(self.radixes), self.num_qudits),
            )

        self._num_params = 0
        self._dim = int(np.prod(self.radixes))

        if initial_value is None:
            self.tensor = np.identity(self.dim, dtype=np.complex128)            
        else:
            if not all((d1==d2 for d1, d2 in zip(self.radixes, initial_value.radixes))):
             raise ValueError(
                f'Expected radixes to be equal between the intial value to desired builder radixes:'
                ' {initail_value.radixes} != {self.radixes}' ,
            )   
            
            self.tensor = initial_value.numpy

        self.tensor = self.tensor.reshape(self.radixes * 2)


    def get_unitary(self, params: RealVector = [], use_jax:bool = False) -> UnitaryMatrix:
        """Build the unitary, see :func:`Unitary.get_unitary` for more."""
        utry = self.tensor.reshape((self.dim, self.dim))
        return UnitaryMatrix(utry, self.radixes, False, use_jax=use_jax)

    def apply_right(
        self,
        utry: UnitaryMatrix,
        location: CircuitLocationLike,
        inverse: bool = False,
        check_arguments: bool = True,
        use_jax: bool = False,
    ) -> None:
        """
        Apply the specified unitary on the right of this UnitaryBuilder.

        ..
                 .-----.   .------.
              n -|     |---|      |-
            n+1 -|     |---| utry |-
                 .     .   '------'
                 .     .
                 .     .
           2n-1 -|     |------------
                 '-----'

        Args:
            utry (UnitaryMatrix): The unitary to apply.

            location (CircuitLocationLike): The qudits to apply the unitary on.

            inverse (bool): If true, apply the inverse of the unitary.

            check_arguments (bool): If true, check the inputs for type and
                value errors.

        Raises:
            ValueError: If `utry`'s size does not match the given location.

            ValueError: if `utry`'s radixes does not match the given location.

        Notes:
            - Applying the unitary on the right is equivalent to multiplying
              the unitary on the left of the tensor. The notation comes
              from the quantum circuit perspective.

            - This operation is performed using tensor contraction.
        """

        if check_arguments:
            if not isinstance(utry, UnitaryMatrix):
                raise TypeError('Expected UnitaryMatrix, got %s', type(utry))

            if not CircuitLocation.is_location(location, self.num_qudits):
                raise TypeError('Invalid location.')

            location = CircuitLocation(location)

            if len(location) != utry.num_qudits:
                raise ValueError('Unitary and location size mismatch.')

            for utry_radix, bldr_radix_idx in zip(utry.radixes, location):
                if utry_radix != self.radixes[bldr_radix_idx]:
                    raise ValueError('Unitary and location radix mismatch.')
        
        location = cast(CircuitLocation, location)
        utry_tensor = utry.get_tensor_format()
        utry_size  = len(utry.radixes)
 
        if inverse:
            offset = 0
            utry_tensor = utry_tensor.conj()
        else:
            offset = utry_size
            
        utry_tensor_indexs    = [i for i in range(2*utry_size)]        
        utry_builder_tensor_indexs = [2*utry_size + i  for i in range(2*self.num_qudits)]        
        output_tensor_index   = [2*utry_size + i  for i in range(2*self.num_qudits)]
        
        for i, loc in enumerate(location):
            utry_builder_tensor_indexs[loc] = offset + i
            output_tensor_index[loc] = (utry_size - offset) + i
        
        if not use_jax:
            self.tensor  = np.einsum(utry_tensor, utry_tensor_indexs, self.tensor, utry_builder_tensor_indexs, output_tensor_index)
        else:
            self.tensor  = jnp.einsum(utry_tensor, utry_tensor_indexs, self.tensor, utry_builder_tensor_indexs, output_tensor_index)


    def apply_left(
        self,
        utry: UnitaryMatrix,
        location: CircuitLocationLike,
        inverse: bool = False,
        check_arguments: bool = True,
        use_jax: bool = False,
    ) -> None:
        """
        Apply the specified unitary on the left of this UnitaryBuilder.

        ..
                 .------.   .-----.
              2 -|      |---|     |-
              3 -| gate |---|     |-
                 '------'   .     .
                            .     .
                            .     .
           2n-1 ------------|     |-
                            '-----'

        Args:
            utry (UnitaryMatrix): The unitary to apply.

            location (CircuitLocationLike): The qudits to apply the unitary on.

            inverse (bool): If true, apply the inverse of the unitary.

            check_arguments (bool): If true, check the inputs for type and
                value errors.

        Raises:
            ValueError: If `utry`'s size does not match the given location.

            ValueError: if `utry`'s radixes does not match the given location.

        Notes:
            - Applying the unitary on the left is equivalent to multiplying
              the unitary on the right of the tensor. The notation comes
              from the quantum circuit perspective.

            - This operation is performed using tensor contraction.
        """

        if check_arguments:
            if not isinstance(utry, UnitaryMatrix):
                raise TypeError('Expected UnitaryMatrix, got %s', type(utry))

            if not CircuitLocation.is_location(location, self.num_qudits):
                raise TypeError('Invalid location.')

            location = CircuitLocation(location)

            if len(location) != utry.num_qudits:
                raise ValueError('Unitary and location size mismatch.')

            for utry_radix, bldr_radix_idx in zip(utry.radixes, location):
                if utry_radix != self.radixes[bldr_radix_idx]:
                    raise ValueError('Unitary and location radix mismatch.')

        location = cast(CircuitLocation, location)
        utry_tensor = utry.get_tensor_format()
        utry_size  = len(utry.radixes)
 
        if inverse:
            offset = utry_size
            utry_tensor = utry_tensor.conj()
        else:
            offset = 0
            
        utry_tensor_indexs          = [i for i in range(2*utry_size)]        
        utry_builder_tensor_indexs  = [2*utry_size + i  for i in range(2*self.num_qudits)]        
        output_tensor_index         = [2*utry_size + i  for i in range(2*self.num_qudits)]
        
        for i, loc in enumerate(location):
            utry_builder_tensor_indexs[self.num_qudits + loc]   = offset + i
            output_tensor_index[self.num_qudits + loc]          = (utry_size - offset) + i
        
        if not use_jax:
            self.tensor  = np.einsum(utry_tensor, utry_tensor_indexs, self.tensor, utry_builder_tensor_indexs, output_tensor_index)
        else:
            self.tensor  = jnp.einsum(utry_tensor, utry_tensor_indexs, self.tensor, utry_builder_tensor_indexs, output_tensor_index)

    def calc_env_matrix(
            self, location: Sequence[int], use_jax: bool = False
    ) :
        """
        Calculates the environment matrix w.r.t. the specified location.

        Args:
            location (Sequence[int]): Calculate the environment matrix with
                respect to the qudit indices in location.

        Returns:
            np.ndarray: The environmental matrix.
        """

        contraction_indexs = list(range(self.num_qudits))+list(range(self.num_qudits))
        for i, loc in enumerate(location):            
            contraction_indexs[loc+self.num_qudits] = self.num_qudits + i + 1

        contraction_indexs_str = "".join([chr(ord('a')+i) for i in contraction_indexs])

        if not use_jax:
            env_tensor = np.einsum(contraction_indexs_str, self.tensor)
        else:
            env_tensor = jnp.einsum(contraction_indexs_str, self.tensor)

        env_mat = env_tensor.reshape((2**len(location), -1))

        return env_mat


    def  _tree_flatten(self):
        children = (self.get_unitary(),)  # arrays / dynamic values
        aux_data = {'radixes': self._radixs,
                    'num_qudits': self.num_qudits
                    
                    }  # static values
        return (children, aux_data)


    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


jax.tree_util.register_pytree_node(UnitaryBuilder,
                               UnitaryBuilder._tree_flatten,
                               UnitaryBuilder._tree_unflatten)
