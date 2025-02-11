# BQSKit Fork - PAM Variants

This fork of BQSKit implements variants of Permutation-Aware Mapping (PAM) [1]. 

[1] J. Liu, E. Younis, M. Weiden, P. Hovland, J. Kubiatowicz and C. Iancu, 
"Tackling the Qubit Mapping Problem with Permutation-Aware Synthesis," 2023 IEEE 
International Conference on Quantum Computing and Engineering (QCE), Bellevue, WA, 
USA, 2023, pp. 745-756, doi: 10.1109/QCE57702.2023.00090.

## Installation

1. Set up a python virtual environment.
2. Install BQSKit using the [Github instructions](https://github.com/BQSKit/bqskit),
   to get its dependencies.
3. Clone this repo and perform an editable install of it.

```sh
python -m venv /path/to/venv

source /path/to/venv/bin/activate

pip install bqskit

git clone git@github.com:m-tremba/bqskit.git

pip install -e /path/to/repo
```

## Basic Usage

Variants are implemented in different branches of the repo. To change variants, change
branches using git. 

```sh
git checkout branch_name
```

To compile circuits using the variants, try building a workflow using the default
PAM workflow builder function included in `bqskit.compiler.compile`{:python}.

```python
from bqskit import Circuit
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.workflow import Workflow
from bqskit.compiler.compile import build_seqpam_mapping_optimization_workflow

# Load a circuit from QASM
circuit = Circuit.from_file("input.qasm")

# Create PAM variant workflow
workflow = build_seqpam_mapping_optimization_workflow()

# Compile the circuit
with Compiler() as compiler:
  compiled_circuit = compiler.compile(circuit.copy(), workflow)

# Save output as QASM
compiled_circuit.save("output.qasm")
```

## Other Information

For all other information, including citations, licenses, and copyright,
see the parent [BQSKit repo](https://github.com/BQSKit/bqskit),
