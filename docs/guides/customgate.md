# Implementing a Custom Gate

BQSKit's claims great portability, and as such, most algorithms in BQSKit can
work natively with any gate set. We have included many commonly used gates
inside of the [`bqskit.ir.gates`](https://bqskit.readthedocs.io/en/latest/source/ir.html#module-bqskit.ir.gates)
subpackage, but you may want to experiment with your own gates. In this tutorial,
we will implement a custom gate in BQSKit. Since BQSKit's algorithms are built
on numerical instantiation, this process is usually as simple as defining a new
subclass with a unitary at a high-level.

For example, let's look at the `TGate` definition in BQSKit:

```python
...
class TGate(ConstantGate, QubitGate):
    _num_qudits = 1
    _qasm_name = 't'
    _utry = UnitaryMatrix(
        [
            [1, 0],
            [0, cmath.exp(1j * cmath.pi / 4)],
        ],
    )
```

A gate is defined by subclassing [`Gate`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.ir.Gate.html#bqskit.ir.Gate),
however, there are some abstract subclasses that can extended instead to simplify the process. For example, the `TGate` is a subclass of
[`ConstantGate`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.ir.ConstantGate.html#bqskit.ir.ConstantGate) and
[`QubitGate`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.ir.QubitGate.html#bqskit.ir.QubitGate). The `ConstantGate`
subclass is used for gates that have a fixed unitary matrix, and the `QubitGate` subclass is used for gates that act only on qubits -- rather than qudits. In the following sections, the process of defining a custom gate will be explained in more detail.

## Defining a Custom Gate

To define a custom gate, you need to subclass [`Gate`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.ir.Gate.html#bqskit.ir.Gate), and
define all the required attributes. These attributes can be defined as instance variables, class variables, or through methods. The following
attributes are required:

- `_num_qudits`: The number of qudits the gate acts on.
- `_num_params`: The number of parameters the gate takes.
- `_radixes`: The radixes of the qudits this gate acts on. This is a list of integers, where each integer is the radix of the corresponding qudit. For example, `[2, 2]` would be a 2-qubit gate, `[3, 3]` would be a 2-qutrit gate, and `[2, 3, 3]` would be a gate that acts on a qubit and two qutrits.
- `_name`: The name of the gate. This used during print operations.
- `_qasm_name`: The name of the gate in QASM. (Qubit only gates, should use lowercase, optional)

Additionally, you will need to override the abstract method [`get_unitary`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.qis.Unitary.get_unitary.html#bqskit.qis.Unitary.get_unitary). This method maps the parameters of the gate to a unitary matrix.

Here is an example of a custom gate that acts on a single qubit:

```python
import cmath
from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.qis.unitary.unitary import RealVector

class MyGate(Gate):
    _num_qudits = 1
    _num_params = 1
    _radixes = (2,)
    _name = 'MyGate'
    _qasm_name = 'mygate'

    def get_unitary(self, params: RealVector) -> UnitaryMatrix:
        theta = params[0]
        return UnitaryMatrix(
            [
                [cmath.exp(1j * theta / 2), 0],
                [0, cmath.exp(-1j * theta / 2)],
            ],
        )
```

Note that the `params` argument is a [`RealVector`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.qis.RealVector.html#bqskit.qis.RealVector) object, which is an alias for many types of float arrays. There is a helper method in the `Gate` class hierarchy called [`check_parameters`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.qis.Unitary.check_parameters.html#bqskit.qis.Unitary.check_parameters) that can be used to validate the parameters before using them. This will check for the correct types and lengths of the parameters:

```python
...
    def get_unitary(self, params: RealVector) -> UnitaryMatrix:
        self.check_parameters(params)
        ...
        return UnitaryMatrix(
            ...
        )
```

As mentioned previously, the required attributes can be defined as class variables, like in the above example, or as instance variables. The following example shows how to define the same gate with instance variables:

```python
import cmath
from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.qis.unitary.unitary import RealVector

class MyGate(Gate):
    def __init__(self, num_qudits: int) -> None:
        self._num_qudits = num_qudits
        self._num_params = 1
        self._radixes = tuple([2] * num_qudits)
        self._name = 'MyGate'

    def get_unitary(self, params: RealVector) -> UnitaryMatrix:
        self.check_parameters(params)
        theta = params[0]
        base = UnitaryMatrix(
            [
                [cmath.exp(1j * theta / 2), 0],
                [0, cmath.exp(-1j * theta / 2)],
            ],
        )
        base.otimes(*[base] * (self._num_qudits - 1)) # base tensor product with itself
```

This style is helpful when the gate's attributes are dependent on the constructor arguments. The `get_unitary` method should be implemented in the same way as before.


## Utilizing Helper Classes

BQSKit provides some helper classes to simplify the process of defining gates. In the first example of this guide, we used the `ConstantGate` and `QubitGate` helper classes. To use these helper subclasses, we will subclass them instead of `Gate`. The following are the current helper classes available:

- [`ConstantGate`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.ir.ConstantGate.html#bqskit.ir.ConstantGate): A gate that has a fixed unitary matrix with no parameters. This will automatically set `_num_params` to 0, and swap the `get_unitary` method for a `_utry` variable. Additionally, these gates are made to be differentiable trivially.
- [`QubitGate`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.ir.QubitGate.html#bqskit.ir.QubitGate): A gate that acts only on qubits. This defines `_radixes` to be all `2`s.
- [`QutritGate`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.ir.QutritGate.html#bqskit.ir.QutritGate): A gate that acts on qutrits. This defines `_radixes` to be all `3`s.
- [`QuditGate`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.ir.QuditGate.html#bqskit.ir.QuditGate): A gate that acts on qudits of the same radix. This swaps the `_radixes` requirement for a `_radix` requirement. This is useful for gates that act on qudits of the same radix, but not necessarily only qubits or qutrits.
- [`ComposedGate`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.ir.ComposedGate.html#bqskit.ir.ComposedGate): A gate that is composed of other gates. This provides methods to dynamically determine if the gate is differentiable or optimizable via other means.

## Differentiable Gates

If you are implementing a parameterized gate, you may want to make it differentiable. By making a gate differentiable, you allow it to be used by out instantiation engine. In turn, this allows synthesis and other algorithms to work more easily with these gates. To do this, you will need to additionally subclass [`DifferentiableUnitary`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.qis.DifferentiableUnitary.html) and implement the [`get_grad`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.qis.DifferentiableUnitary.get_grad.html#bqskit.qis.DifferentiableUnitary.get_grad) method. `ConstantGate`s are trivially differentiable, as they have no parameters.

Most of the time, the [`get_unitary_and_grad`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.qis.DifferentiableUnitary.get_unitary_and_grad.html#bqskit.qis.DifferentiableUnitary.get_unitary_and_grad) method is called by other parts of BQSKit, since both the unitary and gradient are typically needed at the same time. For most gates, computing them at the same time can allow for greater efficiency, since the unitary and gradient can share some computations.

Let's make `MyGate` differentiable:

```python
import cmath
from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.differentiableunitary import DifferentiableUnitary

class MyGate(Gate, DifferentiableUnitary):
    _num_qudits = 1
    _num_params = 1
    _radixes = (2,)
    _name = 'MyGate'
    _qasm_name = 'mygate'

    def get_unitary(self, params: RealVector) -> UnitaryMatrix:
        self.check_parameters(params)
        theta = params[0]
        return UnitaryMatrix(
            [
                [cmath.exp(1j * theta / 2), 0],
                [0, cmath.exp(-1j * theta / 2)],
            ],
        )

    def get_grad(self, params: RealVector) -> npt.NDArray[np.complex128]:
        self.check_parameters(params)
        theta = params[0]
        return np.array(
            [
                [
                    [1j / 2 * cmath.exp(1j * theta / 2), 0],
                    [0, -1j / 2 * cmath.exp(-1j * theta / 2)],
                ],
            ],
        )
```

The `get_grad` method should return a 3D array, where the first index is the parameter index. `get_grad(params)[i]` should return the gradient of the unitary with respect to the `i`-th parameter. The gradient should be a matrix of the same shape as the unitary matrix, where each element is the derivative of the unitary matrix element with respect to the parameter.
