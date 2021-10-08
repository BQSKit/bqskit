# Getting Started

BQSKit allows you to use high-quality synthesis algorithms with ease to
compile quantum circuits. It is designed to be highly customizable to find
a compiler-flow that suites your needs well. In this guide, we will cover
installation and some basic workflows, but there is no one-size-fits-all
when it comes to synthesis, and we recommend you explore the package
when you get more comfortable with it.

## Installation

BQSKit is available for Python 3.7+ on Linux, MacOS, and Windows. BQSKit
and its dependencies are listed on the [Python Package Index](https://pypi.org),
and as such, can be installed with pip:

```sh
pip install bqskit
```

## Basic Usage

BQSKit can be used as a quantum compiler and so much more. Below we have
examples for the common use cases in BQSKit.

### Circuit Optimization Example

The most obvious use for an optimizing quantum compiler is to optimize
a quantum program. Here we can optimize a program given as a circuit in
qasm:

```python
from bqskit.compiler import Compiler
from bqskit.compiler import CompilationTask
from bqskit.ir import Circuit

# Load a circuit from QASM
circuit = Circuit.from_file("tfim.qasm")

# Create a standard CompilationTask to optimize the circuit
task = CompilationTask.optimize(circuit)

# Spawn a compiler and compile the task
with Compiler() as compiler:
    optimized_circuit = compiler.compile(task)
```

We first load the qasm program into a `Circuit` object. Then we create
a `CompilationTask` object, which encapsulates an entire compilation flow.
This includes the circuit to be compiled, as well as, the configured
algorithms that are desired to be used to compile the program.

We provide some default constructors for the common cases, such as
`CompilationTask.optimize` and the ones you will see in the following
sections. However good these default configurations may be, they will be
far from the best; they aim to be widely applicable. The best compilation
flow for your use case will likely require more exploration. Once you are
familiar with the basics, we recommend you explore how these defaults are
built and toy with them.

The last block of the program spawns a `Compiler` object which is responsible
for executing the `CompilationTask`. This will handle the execution efficiently.

### Unitary Synthesis Example

A compiler built around the concept of synthesis should definitely
support synthesis. In fact, the BQSKit compiler does support a variety
of ways to perform synthesis. The following example uses the default
flow for synthesizing a toffoli unitary:

```python
import numpy as np
from bqskit.compiler import Compiler
from bqskit.compiler import CompilationTask

# Construct the unitary as an NumPy array
toffoli = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
])

# Create a standard synthesis CompilationTask
task = CompilationTask.synthesize(toffoli)

# Spawn a compiler and compile the task
with Compiler() as compiler:
    synthesized_circuit = compiler.compile(task)
```

### Circuit Template Instantiation Example

At the core of all our of synthesis tools, is the circuit instantiate
primitive. Circuit Instantiation is the problem where given a circuit
template and a target unitary, one has to find the optimal parameters
for the template such that the distance between the resulting circuit's
operation and the target unitary is minimized. This is according to some
cost function.


## Verifying Results
## Further Reading


Here is an example of

It is composed of two parts:

1. BQSKIT IR: One main difference between other quantum python SDKs is that in BQSKit
all circuits and gates are treated more like a function than a fixed object.

2. BQSKIT Compiler Infrastructure
