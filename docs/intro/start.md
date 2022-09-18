# Getting Started

The Berkeley Quantum Synthesis Toolkit (BQSKit) \[bis • kit\] is a powerful and portable quantum compiler framework.
It can be used with ease to compile quantum programs to efficient physical circuits for any QPU.

## Installation

BQSKit is available for Python 3.8+ on Linux, macOS, and Windows. BQSKit
and its dependencies are listed on the [Python Package Index](https://pypi.org),
and as such, pip can easily install it:

```sh
pip install bqskit
```

An optional extension includes pre-built support for many quantum devices and modules for working with other quantum frameworks. Pip can install this extension by appending `[ext]` to the previous command:

```sh
pip install 'bqskit[ext]'
```

**Note**: If you are using a device with Apple Silicon, you will need to install BQSKit with Anaconda. See the [instructions here](https://github.com/BQSKit/bqskit-tutorial/blob/master/README.md) for more information. We are currently in the process of implementing native ARM support. When that is ready, we will update this note.


## Basic Usage

A standard workflow utilizing BQSKit consists of loading a program into the framework, modeling the target QPU, compiling the program, and exporting the resulting circuit. The below example uses BQSKit to optimize an input circuit provided by a qasm file:

```python
from bqskit import compile, Circuit

# Load a circuit from QASM
circuit = Circuit.from_file("input.qasm")

# Compile the circuit
compiled_circuit = compile(circuit)

# Save output as QASM
compiled_circuit.save("output.qasm")
```

To learn more about BQSKit, follow the [tutorial series](https://github.com/BQSKit/bqskit-tutorial/blob/master/1_comping_with_bqskit.ipynb) or refer to the [documentation](https://bqskit.readthedocs.io/en/latest/).
