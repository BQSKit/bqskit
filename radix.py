# Modify the __init__ method in the UnitaryMatrix class

def __init__(
    self,
    input: UnitaryLike,
    radixes: Union[int, Sequence[int]] = [],
    check_arguments: bool = True,
) -> None:
    """
    Constructs a `UnitaryMatrix` from the supplied unitary matrix.

    Args:
        input (UnitaryLike): The unitary matrix input.

        radixes (int | Sequence[int]): A single radix or a sequence with its length equal to
            the number of qudits this `UnitaryMatrix` can act on. Each
            element specifies the base, number of orthogonal states,
            for the corresponding qudit. By default, the constructor
            will attempt to calculate `radixes` from `utry`.

        check_arguments (bool): If true, check arguments for type
            and value errors.

    Raises:
        ValueError: If `input` is not unitary.

        ValueError: If the dimension of `input` does not match the
            expected dimension from `radixes`.

        RuntimeError: If `radixes` is not specified and the
            constructor cannot infer it.
    """

    # Copy constructor
    if isinstance(input, UnitaryMatrix):
        self._utry = input.numpy
        self._radixes = input.radixes
        self._dim = input.dim
        return

    if check_arguments and not is_square_matrix(input):
        raise TypeError(f'Expected square matrix, got {type(input)}.')

    if check_arguments and not UnitaryMatrix.is_unitary(input):
        raise ValueError('Input failed unitary condition.')

    dim = len(input)

    # Calculate radixes if not explicitly provided
    if isinstance(radixes, int):
        # If a single radix is provided, broadcast it for the whole system
        self._radixes = (radixes,) * int(np.round(np.log2(dim)))

    elif isinstance(radixes, Sequence):
        if len(radixes) != dim:
            raise ValueError(
                'Length of radixes sequence does not match dimension of input matrix.'
            )
        self._radixes = tuple(radixes)

    else:
        raise TypeError('Invalid type for radixes parameter.')

    if check_arguments and not is_valid_radixes(self.radixes):
        raise TypeError('Invalid qudit radixes.')

    if check_arguments and np.prod(self.radixes) != dim:
        raise ValueError('Qudit radixes mismatch with dimension.')

    self._utry = np.array(input, dtype=np.complex128)
    self._dim = dim
