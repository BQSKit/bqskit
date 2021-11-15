# Berkeley Quantum Synthesis Toolkit (BQSKit)

BQSKit allows you to use high-quality synthesis algorithms with ease to
compile quantum circuits. It is designed to be highly customizable to find
a compiler-flow that suites your needs well.

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
if __name__ == '__main__':
    with Compiler() as compiler:
        optimized_circuit = compiler.compile(task)
```

We first load the qasm program into a `Circuit` object. Then we create
a `CompilationTask` object, which encapsulates an entire compilation flow.
This includes the circuit to be compiled, as well as, the configured
algorithms that are desired to be used to compile the program.

We provide some default constructors for the common cases, such as
`CompilationTask.optimize` and the ones you will see in the following
sections. These these default configurations aim to be widely applicable,
and as a result, may be far from the best. The best compilation
flow for your use case will likely require some experimentation. Once you are
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
if __name__ == '__main__':
    with Compiler() as compiler:
        synthesized_circuit = compiler.compile(task)
```

## Verifying Results

All of BQSKit's synthesis algorithms are approximate, which mean the
compiled results might be slightly different than what was inputted.
For most cases, these differences are close to floating point thresholds,
but can be greater in magnitude. Most algorithms will offer some sort
of control over this error, but it is important to be able to measure it.

For simulatable circuits, we can compare the unitaries directly:

```python
dist = synthesized_circuit.get_unitary().get_distance_from(toffoli)
assert dist < 1e-10
```

## Copyright

Berkeley Quantum Synthesis Toolkit (BQSKit) Copyright (c) 2021,
The Regents of the University of California, through Lawrence
Berkeley National Laboratory (subject to receipt of any required
approvals from the U.S. Dept. of Energy) and Massachusetts
Institute of Technology (MIT).  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.
