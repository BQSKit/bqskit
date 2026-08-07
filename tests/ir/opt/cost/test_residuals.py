from __future__ import annotations

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.opt import HilbertSchmidtResidualsGenerator


def test_hilbert_schmidt_residuals(r3_qubit_circuit: Circuit) -> None:
    x0 = np.random.random((r3_qubit_circuit.num_params,))
    cost = HilbertSchmidtResidualsGenerator().gen_cost(
        r3_qubit_circuit, r3_qubit_circuit.get_unitary(x0),
    )
    assert cost.get_cost(x0) < 1e-10
