# Berkeley Quantum Synthesis Toolkit (BQSKit)

The Berkeley Quantum Synthesis Toolkit (BQSKit) \[bis • kit\] is a powerful
and portable quantum compiler framework. It can be used with ease to compile
quantum programs to efficient physical circuits for any QPU.

## Installation

BQSKit is available for Python 3.8+ on Linux, macOS, and Windows. BQSKit
and its dependencies are listed on the [Python Package Index](https://pypi.org),
and as such, pip can easily install it:

```sh
pip install bqskit
```

## Basic Usage

A standard BQSKit workflow loads a program into the framework, models the
target QPU, compiles the program, and exports the resulting circuit. The
below example uses BQSKit to optimize an input circuit provided by a qasm
file:

```python
from bqskit import compile, Circuit

# Load a circuit from QASM
circuit = Circuit.from_file("input.qasm")

# Compile the circuit
compiled_circuit = compile(circuit)

# Save output as QASM
compiled_circuit.save("output.qasm")
```

To learn more about BQSKit, follow the
[tutorial series](https://github.com/BQSKit/bqskit-tutorial/) or refer to
the [documentation](https://bqskit.readthedocs.io/en/latest/).

## How to Cite

You can use the [software disclosure](https://www.osti.gov/biblio/1785933)
to cite the BQSKit package.

Additionally, if you used or extended a specific algorithm, you should cite
that individually. BQSKit passes will include a relevant reference in
their documentation.

## License

The software in this repository is licensed under a **BSD free software
license** and can be used in source or binary form for any purpose as long
as the simple licensing requirements are followed. See the
**[LICENSE](https://github.com/BQSKit/bqskit/blob/master/LICENSE)** file
for more information.

## Copyright

Berkeley Quantum Synthesis Toolkit (BQSKit) Copyright (c) 2021,
The Regents of the University of California, through Lawrence
Berkeley National Laboratory (subject to receipt of any required
approvals from the U.S. Dept. of Energy) and Massachusetts
Institute of Technology (MIT).  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to
do so.
