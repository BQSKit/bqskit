from __future__ import annotations

import numpy as np

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler

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
        print(synthesized_circuit)
